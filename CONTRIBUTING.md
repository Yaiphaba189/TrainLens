# Contributing to NeuroLens

Thanks for helping. Bug reports, examples and pull requests are all welcome.

## Set up

You need Python 3.9+ and a C++17 compiler: Xcode command line tools on
macOS, or `build-essential` on Linux. pip downloads CMake, and SDL2 too if
your system doesn't have it.

```sh
git clone https://github.com/Yaiphaba189/TrainLens
cd TrainLens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"        # builds the viewer and installs `neurolens`
```

To use your system SDL2, which gives faster builds, install it first:
`brew install sdl2` on macOS, or `sudo apt install libsdl2-dev libgl1-mesa-dev zlib1g-dev` on Debian or Ubuntu.

## Run the tests

```sh
neurolens --selftest                    # C++ checks: protocol, forward pass, training alerts
neurolens --check examples/xor.nlens    # file loading
pytest                                  # Python client; PyTorch tests run when torch is installed
```

When you change the viewer, also look at it. These commands save a screenshot
without you touching the window:

```sh
neurolens --frames 150 --screenshot /tmp/shot.png            # network demo
python examples/torch_monitor_cnn.py & neurolens --frames 900 --screenshot /tmp/monitor.png
```

## Build only the viewer (no pip)

```sh
cmake -B build && cmake --build build -j
NEUROLENS_VIEWER=$PWD/build/neurolens-viewer python examples/numpy_xor_train.py
```

`NEUROLENS_VIEWER` tells the Python package where your viewer build is.

## Layout

| Path | What lives there |
|---|---|
| `viewer/` | C++ viewer: window and input (`main.cpp`), drawing (`render.cpp`), protocol, TCP server, training checks |
| `src/neurolens/` | Python package: `Viewer` client, `monitor()`, `watch()`, `launch()`, `.nlens` writer, CLI |
| `tests/` | pytest suite |
| `examples/` | runnable demos |
| `CMakeLists.txt`, `pyproject.toml` | build: scikit-build-core compiles the viewer into the wheel |

## Guidelines

- Keep the client non-blocking. Nothing in `neurolens` may slow down or
  crash a training loop when the viewer is closed or slow.
- Protocol changes need three things: a parser in `viewer/protocol.cpp`, a
  sender in `client.py`, and a row in the README protocol table. Keep old
  commands working.
- A new automatic check goes in `viewer/training.cpp`. It needs a self-test
  in `run_selftest()` and a row in the README checks table, and its message
  must say what to do about the problem.
- Match the surrounding style. C++17 with no extra dependencies beyond SDL2,
  OpenGL and zlib. Python with no required dependencies; torch is imported
  only inside the functions that need it.

## Help wanted

- **Windows support:** `viewer/server.cpp` uses POSIX sockets and needs a Winsock path.
- **More layer types in the network view:** conv feature maps, attention.
- **Other frameworks:** JAX and TensorFlow helpers like `monitor()`.

## Releasing (maintainers)

1. Update `version` in `pyproject.toml` and the CHANGELOG.
2. Tag the release: `git tag v0.1.0 && git push --tags`.
3. The `release` workflow builds wheels for macOS and Linux plus an sdist,
   and publishes them to PyPI with trusted publishing. Before the first
   release, set up the trusted publisher on PyPI for this repository, with
   workflow `release.yml` and environment `pypi`.
