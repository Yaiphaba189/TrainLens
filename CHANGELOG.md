# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First public release.

### Added
- `train-lens`: C++/OpenGL viewer that is compiled into the wheel, so it needs no system libraries on macOS or Linux.
- Network view: neurons and connections lit by real activations and weights, with flat and 3D layouts, hover values, muting neurons by clicking, and PNG screenshots.
- Training monitor for any PyTorch model (`train_lens.monitor`): loss and metric curves, per-layer gradient norms and update sizes.
- Automatic checks: NaN or Inf loss, loss spikes, divergence, plateaus, overfitting, exploding or missing gradients, learning rate too high or too low, and dead ReLUs.
- `train_lens.watch` streams the live activations of a PyTorch MLP.
- `train_lens.Viewer` sends data from any framework; `train_lens.launch()` opens the window from Python.
- A plain-text protocol shared by `.tlens` files and the TCP stream.
