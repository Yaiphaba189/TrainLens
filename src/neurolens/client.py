"""Streams a network to a running NeuroLens viewer over TCP.

Every call returns immediately. Messages go out on a background thread; if
the viewer is not running or falls behind, live data (activations, inputs,
status, layer stats) is dropped rather than slowing down your training loop.
Topology, weights and metric curves are remembered and re-sent whenever the
connection comes back, so a viewer opened mid-run still shows the whole run.
"""

import queue
import socket
import sys
import threading
import time

__all__ = ["Viewer"]


def _cpu(x):
    """Copies a tensor or array off the GPU and out of autograd, cheaply."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().float().cpu().numpy().copy()
    if hasattr(x, "copy"):
        return x.copy()
    return list(x)


def _flat(x):
    """Flattens a tensor, numpy array or nested list into a list of floats."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    if hasattr(x, "reshape") and hasattr(x, "tolist"):
        return [float(v) for v in x.reshape(-1).tolist()]
    out = []
    for v in x:
        if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
            out.extend(_flat(v))
        else:
            out.append(float(v))
    return out


def _nums(x):
    return " ".join(format(v, ".6g") for v in _flat(x))


def _value(v):
    return format(v, ".4g") if isinstance(v, float) else str(v)


class Viewer:
    """Connection to a NeuroLens viewer window.

    >>> v = Viewer()                                   # 127.0.0.1:5757
    >>> v.topology([2, 8, 1], ["tanh", "sigmoid"])
    >>> v.weights(0, W0, b0); v.weights(1, W1, b1)     # shapes (out, in), (out,)
    >>> v.input([0.0, 1.0])                            # viewer runs the forward pass
    >>> v.frame([x, h, y])                             # or send real activations
    >>> v.status(epoch=3, loss=0.12)
    """

    def __init__(self, host="127.0.0.1", port=5757, queue_size=4, retry_every=2.0, quiet=False,
                 max_history=20000):
        self.host, self.port = host, port
        self._retry = retry_every
        self._quiet = quiet
        self._q = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._header = []    # layers / act / labels lines
        self._params = {}    # (layer, kind) -> values, formatted on the sender thread
        self._dirty = False  # header or params changed since last sent
        self._history = []   # every metric line, replayed to a viewer that connects later
        self._pending = []   # metric lines not yet sent on this connection
        self._replay = False
        self._max_history = max_history
        self._closed = False
        self._connected = False
        self._warned = False
        self._thread = threading.Thread(target=self._run, name="neurolens", daemon=True)
        self._thread.start()

    @property
    def connected(self):
        return self._connected

    # ---- network description (kept and re-sent on reconnect) ----

    def topology(self, sizes, acts=None, labels=None):
        """Layer sizes from input to output, one activation name per
        non-input layer (relu, tanh, sigmoid, softmax, linear), and optional
        names for the output neurons."""
        sizes = [int(s) for s in sizes]
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        lines = ["layers " + " ".join(map(str, sizes))]
        if acts is not None:
            if len(acts) != len(sizes) - 1:
                raise ValueError(f"need {len(sizes) - 1} activation names, got {len(acts)}")
            lines.append("act " + " ".join(acts))
        if labels is not None:
            lines.append("labels " + " ".join(str(l).replace(" ", "_") for l in labels))
        with self._lock:
            self._header = lines
            self._params = {}
            self._dirty = True
        return self

    def weights(self, layer, w, b=None):
        """Weights from layer ``layer`` to ``layer + 1``, shaped (out, in)
        like ``torch.nn.Linear.weight``, and optionally its bias."""
        w = _cpu(w)
        b = None if b is None else _cpu(b)
        with self._lock:
            self._params[(int(layer), "w")] = w
            if b is not None:
                self._params[(int(layer), "b")] = b
            self._dirty = True

    # ---- training curves (never dropped; replayed on reconnect) ----

    def metric(self, name, value, step=None):
        """Adds a point to a curve. Names starting with ``val_`` (or ``test_``,
        ``eval_``) are drawn on the same chart as the name without the prefix,
        so ``loss`` and ``val_loss`` share one chart."""
        value = float(value.item() if hasattr(value, "item") else value)
        line = f"metric {str(name).replace(' ', '_')} {format(value, '.7g')}"
        if step is not None:
            line += f" {int(step)}"
        with self._lock:
            self._history.append(line)
            self._pending.append(line)
            if len(self._history) > self._max_history:
                self._history = self._history[::2]  # keep the shape of a very long run

    def metrics(self, step=None, **values):
        """``metrics(step, loss=..., acc=...)``: several curves at one step."""
        for name, value in values.items():
            self.metric(name, value, step)

    def layer_stats(self, rows):
        """Per-layer health: rows of (name, grad_norm, weight_norm, update_ratio).
        Use -1 for unknown, and grad_norm -1 for a layer that got no gradient."""
        lines = [f"pstats {len(rows)}"]
        for i, (name, g, w, u) in enumerate(rows):
            lines.append(f"pstat {i} {str(name).replace(' ', '_') or '?'} {g:.4g} {w:.4g} {u:.4g}")
        self._put("\n".join(lines))

    # ---- live data (dropped when the viewer is slow or absent) ----

    def input(self, x):
        """Sets the input; the viewer computes the rest from the weights."""
        self._put("input " + _nums(x))

    def acts(self, layer, values):
        """Activations of one layer (0 = input)."""
        self._put(f"acts {int(layer)} " + _nums(values))

    def frame(self, layers):
        """Activations for layers 0..n-1 in one message, so they arrive
        together. ``None`` entries are skipped."""
        lines = [f"acts {i} " + _nums(v) for i, v in enumerate(layers) if v is not None]
        if lines:
            self._put("\n".join(lines))

    def status(self, text=None, **values):
        """One line of text shown in the viewer, e.g. ``status(epoch=3, loss=0.1)``."""
        parts = [] if text is None else [str(text)]
        parts += [f"{k}={_value(v)}" for k, v in values.items()]
        self._put("status " + "  ".join(parts))

    def reset(self):
        """Forget sent activations and input; the viewer animates its inputs again."""
        self._put("reset")

    def send(self, line):
        """Sends a raw protocol line (see README.md)."""
        self._put(str(line))

    def close(self, timeout=2.0):
        """Sends what is still queued (best effort) and stops the thread."""
        self._closed = True
        self._thread.join(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- internals ----

    def _put(self, msg):
        if self._closed:
            return
        try:
            self._q.put_nowait(msg)
        except queue.Full:  # keep the newest data
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(msg)
            except queue.Full:
                pass

    def _flush(self, sock):
        """Sends header changes and metric lines that are due, in that order."""
        with self._lock:
            lines = list(self._header) if self._dirty else []
            params = sorted(self._params.items()) if self._dirty else []
            self._dirty = False
            if self._replay:
                metric_lines = ["clearmetrics"] + self._history
                self._replay = False
            else:
                metric_lines = self._pending
            self._pending = []
        for (layer, kind), values in params:  # formatted outside the lock
            lines.append(f"{kind} {layer} " + _nums(values))
        lines += metric_lines
        if lines:
            sock.sendall(("\n".join(lines) + "\n").encode())

    def _connect(self):
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.0)
        except OSError:
            if not self._warned and not self._quiet:
                print(f"neurolens: no viewer at {self.host}:{self.port} yet; run `neurolens` in a terminal or "
                      "call neurolens.launch() (training continues, will keep retrying)", file=sys.stderr)
                self._warned = True
            return None
        s.settimeout(5.0)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._lock:
            self._dirty = True
            self._replay = True
        self._connected = True
        if not self._quiet:
            print(f"neurolens: connected to viewer at {self.host}:{self.port}", file=sys.stderr)
        return s

    def _drop_queued(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _run(self):
        sock = None
        while True:
            if sock is None:
                if self._closed:
                    return
                sock = self._connect()
                if sock is None:
                    self._drop_queued()  # stale live data is not worth replaying
                    deadline = time.monotonic() + self._retry
                    while not self._closed and time.monotonic() < deadline:
                        time.sleep(0.05)
                    continue
            try:
                self._flush(sock)
                try:
                    msg = self._q.get(timeout=0.05)
                except queue.Empty:
                    if self._closed:
                        self._flush(sock)
                        break
                    continue
                self._flush(sock)  # a topology change queued before this message goes first
                sock.sendall((msg + "\n").encode())
            except OSError:
                self._connected = False
                sock.close()
                sock = None
                if not self._quiet:
                    print("neurolens: lost the viewer; retrying", file=sys.stderr)
        sock.close()
        self._connected = False
