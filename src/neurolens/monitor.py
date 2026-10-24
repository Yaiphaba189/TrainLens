"""Training monitor for any PyTorch model: curves, gradient flow, update sizes.

    mon = neurolens.monitor(model, optimizer)
    for step, (x, y) in enumerate(loader):
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mon.step(loss=loss)              # after backward, before the next zero_grad
    mon.log(val_loss=v, val_acc=a)       # epoch-level numbers, sent as they are

Works with CNNs, transformers, LoRA fine-tunes or anything else: it only reads
parameters and their gradients. The viewer then shows the curves, a gradient
and update-size row per layer, and checks such as NaN loss, exploding or
vanishing gradients, a learning rate that looks too high or too low, and
overfitting.
"""

import math
import sys
import time

from .client import Viewer

__all__ = ["Monitor", "monitor"]


def _short(name, limit=40):
    """Keeps the most specific end of a long module path."""
    if len(name) <= limit:
        return name
    parts = name.split(".")
    out = parts[-1]
    for p in reversed(parts[:-1]):
        if len(p) + 1 + len(out) > limit - 2:
            break
        out = p + "." + out
    return ".." + out


class Monitor:
    """See :func:`monitor`."""

    def __init__(self, model, optimizer=None, viewer=None, *, every=0.5, max_layers=40,
                 track_updates=True, max_update_params=200_000_000, host="127.0.0.1", port=5757):
        self.model = model
        self.optimizer = optimizer
        self.viewer = viewer or Viewer(host, port)
        self.every = every
        self._groups = self._make_groups(model, max_layers)
        self._step = 0
        self._sums = {}  # name -> [sum, count] since the last send
        self._last_send = float("-inf")
        self._updates = [-1.0] * len(self._groups)
        self._want_update = False
        self._snapshot = None
        self._hooks = []
        self._warned_no_grad = False

        n_params = sum(p.numel() for _, ps in self._groups for p in ps)
        if track_updates and optimizer is not None:
            if not hasattr(optimizer, "register_step_pre_hook"):
                print("neurolens: this torch version has no optimizer step hooks; update sizes are off",
                      file=sys.stderr)
            elif n_params > max_update_params:
                print(f"neurolens: {n_params:,} trainable values is over max_update_params; update sizes are off "
                      "(they need a copy of the weights for one step)", file=sys.stderr)
            else:
                self._hooks = [optimizer.register_step_pre_hook(self._pre_step),
                               optimizer.register_step_post_hook(self._post_step)]

    @staticmethod
    def _make_groups(model, max_layers):
        """One group per module that owns trainable parameters, merged into at
        most max_layers consecutive buckets for big models."""
        groups = []
        for name, m in model.named_modules():
            ps = [p for p in m.parameters(recurse=False) if p.requires_grad]
            if ps:  # "encoder.0:Conv2d", so bare Sequential indices still say what the layer is
                groups.append((f"{name}:{type(m).__name__}" if name else type(m).__name__, ps))
        if not groups:
            raise ValueError("the model has no trainable parameters")
        if len(groups) <= max_layers:
            return [(_short(n), ps) for n, ps in groups]
        size = math.ceil(len(groups) / max_layers)
        merged = []
        for i in range(0, len(groups), size):
            chunk = groups[i:i + size]
            name = _short(chunk[0][0], 22) + "+" + str(len(chunk) - 1)
            merged.append((name, [p for _, ps in chunk for p in ps]))
        return merged

    # ---- update size: |W_after - W_before| / |W| for one optimizer step ----

    def _pre_step(self, optimizer, args, kwargs):
        if self._want_update:
            self._snapshot = [[p.detach().clone() for p in ps] for _, ps in self._groups]

    def _post_step(self, optimizer, args, kwargs):
        if self._snapshot is None:
            return
        import torch

        with torch.no_grad():
            for i, ((_, ps), old) in enumerate(zip(self._groups, self._snapshot)):
                delta = sum(float((p.float() - o.float()).norm()) ** 2 for p, o in zip(ps, old)) ** 0.5
                weight = sum(float(p.float().norm()) ** 2 for p in ps) ** 0.5
                self._updates[i] = delta / (weight + 1e-12)
        self._snapshot = None
        self._want_update = False

    # ---- public ----

    def step(self, step=None, **metrics):
        """Call once per training step, after backward() and before the next
        zero_grad(). Metrics are averaged between sends (every ``every``
        seconds), which smooths noisy per-batch values."""
        self._step = int(step) if step is not None else self._step + 1
        for name, v in metrics.items():
            v = float(v.item() if hasattr(v, "item") else v)
            s = self._sums.setdefault(name, [0.0, 0])
            s[0] += v
            s[1] += 1
        now = time.monotonic()
        if now - self._last_send < self.every:
            return
        self._last_send = now

        for name, (total, count) in self._sums.items():
            self.viewer.metric(name, total / count, self._step)
        self._sums = {}
        if self.optimizer is not None:
            self.viewer.metric("lr", self.optimizer.param_groups[0]["lr"], self._step)
        self._send_layer_stats()
        self._want_update = bool(self._hooks)

    def log(self, step=None, **metrics):
        """Sends metrics right away, without averaging (validation results, epoch summaries)."""
        self.viewer.metrics(self._step if step is None else step, **metrics)

    def status(self, text=None, **values):
        self.viewer.status(text, **values)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _send_layer_stats(self):
        import torch

        rows, total = [], 0.0
        any_grad = False
        with torch.no_grad():
            for i, (name, ps) in enumerate(self._groups):
                grads = [p.grad for p in ps if p.grad is not None]
                if grads:
                    any_grad = True
                    g = sum(float(x.detach().float().norm()) ** 2 for x in grads) ** 0.5
                    total += g * g
                else:
                    g = -1.0
                w = sum(float(p.detach().float().norm()) ** 2 for p in ps) ** 0.5
                rows.append((name, g, w, self._updates[i]))
        self.viewer.layer_stats(rows)
        if any_grad:
            self.viewer.metric("grad_norm", total ** 0.5, self._step)
        elif not self._warned_no_grad:
            self._warned_no_grad = True
            print("neurolens: no gradients found; call monitor.step() after loss.backward() "
                  "and before optimizer.zero_grad()", file=sys.stderr)


def monitor(model, optimizer=None, viewer=None, **kwargs):
    """Watches training of any PyTorch model.

    optimizer       pass it to get the learning rate curve and per-layer update sizes
    every           seconds between sends (default 0.5)
    max_layers      rows in the layer table; bigger models are grouped
    track_updates   measure |dW|/|W| (copies the weights once per send)
    host, port      where the viewer listens
    """
    return Monitor(model, optimizer, viewer, **kwargs)
