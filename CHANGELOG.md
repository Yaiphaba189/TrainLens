# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-10-24

First release on PyPI, as `neurolens`. The project was renamed from TrainLens
because PyPI considers `train-lens` too similar to the existing `trainlens` package.
Version 0.1.0 was only published as a GitHub release, under the old name.

### Added
- `neurolens`: C++/OpenGL viewer that is compiled into the wheel, so it needs no system libraries on macOS or Linux.
- Network view: neurons and connections lit by real activations and weights, with flat and 3D layouts, hover values, muting neurons by clicking, and PNG screenshots.
- Training monitor for any PyTorch model (`neurolens.monitor`): loss and metric curves, per-layer gradient norms and update sizes.
- Automatic checks: NaN or Inf loss, loss spikes, divergence, plateaus, overfitting, exploding or missing gradients, learning rate too high or too low, and dead ReLUs.
- `neurolens.watch` streams the live activations of a PyTorch MLP.
- `neurolens.Viewer` sends data from any framework; `neurolens.launch()` opens the window from Python.
- A plain-text protocol shared by `.nlens` files and the TCP stream.
