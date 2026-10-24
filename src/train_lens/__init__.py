"""TrainLens: watch neural networks train, live.

    import train_lens
    train_lens.launch()                    # open the viewer window (or run `train-lens` in a terminal)
    mon = train_lens.monitor(model, opt)   # any PyTorch model: curves, gradient flow, checks
    handle = train_lens.watch(model)       # PyTorch MLP: live neurons during training or inference
    viewer = train_lens.Viewer()           # any framework: send metrics / weights / activations yourself
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("train-lens")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0.dev"

from ._viewer import launch, viewer_path
from .client import Viewer
from .tlensfile import save_tlens, save_torch

__all__ = ["Viewer", "launch", "monitor", "watch", "save_tlens", "save_torch", "viewer_path", "__version__"]


def monitor(model, optimizer=None, viewer=None, **kwargs):
    """Watches training of any PyTorch model. See :func:`train_lens.monitor.monitor`."""
    from .monitor import monitor as _monitor  # torch is only needed here

    return _monitor(model, optimizer, viewer, **kwargs)


def watch(model, viewer=None, **kwargs):
    """Streams a PyTorch MLP to the viewer with forward hooks. See :func:`train_lens.torch_watch.watch`."""
    from .torch_watch import watch as _watch  # torch is only needed here

    return _watch(model, viewer, **kwargs)
