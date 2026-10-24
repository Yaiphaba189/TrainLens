"""Attaches NeuroLens to a PyTorch MLP with forward hooks.

    handle = neurolens.watch(model)      # during training or inference
    ...                                 # run the model as usual
    handle.status(epoch=e, loss=l)
    handle.remove()

Only stacks of ``nn.Linear`` layers are shown (an MLP, or the MLP head of a
bigger model). The activations shown are the real ones from your forward
pass, for one sample of the batch.
"""

import time

from .client import Viewer

__all__ = ["watch", "describe", "Watch"]

# Activation modules and the closest curve the viewer knows.
_ACT_NAMES = {
    "ReLU": "relu", "ReLU6": "relu", "LeakyReLU": "relu", "GELU": "relu", "SiLU": "relu",
    "ELU": "relu", "Mish": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid", "Softmax": "softmax",
    "Identity": "linear",
}


def _mlp_layers(model):
    """[(linear, activation module or None), ...] in forward order.

    Walks leaf modules in registration order, so the model must define its
    layers in the order forward() uses them (always true for nn.Sequential).
    """
    import torch.nn as nn

    layers = []
    for m in model.modules():
        if any(True for _ in m.children()):
            continue
        if isinstance(m, nn.Linear):
            layers.append([m, None])
        elif type(m).__name__ in _ACT_NAMES and layers and layers[-1][1] is None:
            layers[-1][1] = m
    if not layers:
        raise ValueError("no nn.Linear layers found in the model")
    for (a, _), (b, _) in zip(layers, layers[1:]):
        if a.out_features != b.in_features:
            raise ValueError(f"layers do not chain ({a} then {b}); NeuroLens shows plain stacks of nn.Linear")
    return [tuple(x) for x in layers]


def describe(model):
    """Returns (layers, sizes, activation names) for a torch MLP."""
    layers = _mlp_layers(model)
    sizes = [layers[0][0].in_features] + [lin.out_features for lin, _ in layers]
    acts = [_ACT_NAMES[type(a).__name__] if a is not None else "linear" for _, a in layers]
    return layers, sizes, acts


class Watch:
    """Handle returned by :func:`watch`."""

    def __init__(self, model, viewer, max_hz, weights_every, sample, labels):
        self._layers, sizes, acts = describe(model)
        self.viewer = viewer
        self._n = len(sizes)
        self._sample = sample
        self._min_dt = 1.0 / max_hz if max_hz else 0.0
        self._weights_every = weights_every
        self._last_frame = float("-inf")
        self._last_weights = float("-inf")
        self._active = False
        self._frame = [None] * self._n

        viewer.topology(sizes, acts, labels)
        self.send_weights()
        first = self._layers[0][0]
        self._handles = [first.register_forward_pre_hook(self._pre)]
        for i, (lin, act) in enumerate(self._layers):
            self._handles.append(lin.register_forward_hook(self._make_hook(i + 1, act)))

    def _row(self, t):
        t = t.detach().reshape(-1, t.shape[-1])
        return t[min(self._sample, t.shape[0] - 1)]

    def _pre(self, module, args):
        now = time.monotonic()
        self._active = now - self._last_frame >= self._min_dt
        if self._active:
            self._last_frame = now
            self._frame = [None] * self._n
            self._frame[0] = self._row(args[0])

    def _make_hook(self, idx, act):
        def hook(module, args, output):
            if not self._active:
                return
            import torch

            with torch.no_grad():
                row = self._row(output)
                if act is not None:  # show the value after the activation
                    row = act(row.unsqueeze(0))[0]
            self._frame[idx] = row
            if idx == self._n - 1:
                self._active = False
                self.viewer.frame(self._frame)
                if self._weights_every is not None and time.monotonic() - self._last_weights >= self._weights_every:
                    self.send_weights()

        return hook

    def send_weights(self):
        for l, (lin, _) in enumerate(self._layers):
            self.viewer.weights(l, lin.weight, lin.bias)
        self._last_weights = time.monotonic()

    def status(self, text=None, **values):
        self.viewer.status(text, **values)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def watch(model, viewer=None, *, max_hz=20, weights_every=1.0, sample=0, labels=None,
          host="127.0.0.1", port=5757):
    """Streams a torch MLP to the viewer while it runs.

    max_hz         how many forward passes per second are shown (others are skipped)
    weights_every  seconds between weight updates; None sends them once
    sample         which item of the batch to show
    labels         names for the output neurons, e.g. class names
    """
    viewer = viewer or Viewer(host, port)
    return Watch(model, viewer, max_hz, weights_every, sample, labels)
