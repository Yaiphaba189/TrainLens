# TrainLens

[![PyPI](https://img.shields.io/pypi/v/train-lens)](https://pypi.org/project/train-lens/)
[![Python](https://img.shields.io/pypi/pyversions/train-lens)](https://pypi.org/project/train-lens/)
[![CI](https://github.com/Yaiphaba189/TrainLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Yaiphaba189/TrainLens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Watch neural networks train, live.** TrainLens is a fast C++/OpenGL
window that shows what your model is doing *while it trains*: loss curves,
gradient flow through every layer, update sizes, and the neurons themselves.
Built-in checks tell you when something goes wrong: NaN loss, exploding
gradients, a learning rate that is too high, overfitting.

```sh
pip install train-lens
```

```python
import train_lens

train_lens.launch()                        # opens the window
mon = train_lens.monitor(model, optimizer) # any PyTorch model

for step, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    mon.step(step, loss=loss)
```

![Training monitor flagging overfitting](https://raw.githubusercontent.com/Yaiphaba189/TrainLens/main/docs/monitor-overfitting.png)

| Live network view | LR far too high: the checks catch it |
|---|---|
| ![Network view](https://raw.githubusercontent.com/Yaiphaba189/TrainLens/main/docs/demo-2d.png) | ![LR too high](https://raw.githubusercontent.com/Yaiphaba189/TrainLens/main/docs/monitor-lr-too-high.png) |

**Contents:** [Quick start](#quick-start) ·
[Training monitor](#training-monitor) · [Network view](#network-view) ·
[Any framework](#any-framework) · [How it works](#how-it-works) ·
[Protocol](#protocol) · [Examples](#examples) ·
[Troubleshooting](#troubleshooting) · [Roadmap](#roadmap) ·
[Contributing](#contributing)

## Why TrainLens

- **Works with any PyTorch model.** CNNs, transformers and LoRA fine-tunes
  all work: `monitor()` only reads parameters and gradients.
- **Tells you what's wrong.** It runs about ten rule-of-thumb checks
  continuously, and each warning says what to try.
- **Never slows training down.** Data goes out on a background thread. If
  the window is closed or busy, data is dropped instead of blocking your loop.
- **Open the window any time.** Curves are replayed from step 0, so you can
  start the viewer halfway through a run.
- **No setup.** The viewer is compiled into the wheel. There is nothing to
  build, no browser, and no server.
- **Any language.** It uses a tiny plain-text TCP protocol, so a few lines
  of code are enough to send data from JAX, TensorFlow, C++ or a shell script.

## Quick start

1. **Install** on macOS (Intel or Apple Silicon) or Linux, with Python 3.9+:
   ```sh
   pip install train-lens              # add [torch] to also install PyTorch
   ```
2. **Add three lines** to your training script:
   ```python
   import train_lens
   train_lens.launch()                          # once, before training
   mon = train_lens.monitor(model, optimizer)   # once, before training
   ...
   mon.step(step, loss=loss)                    # every step, after optimizer.step()
   ```
3. **Run your script as usual.** The window opens, and curves appear within a second.

Windows is not supported yet ([help wanted](CONTRIBUTING.md#help-wanted)).
Platforms without a prebuilt wheel compile the viewer during `pip install`.
That needs a C++ compiler, plus OpenGL and X11 headers on Linux.

## Start the viewer

From a terminal:

```sh
train-lens                  # opens the window and waits for your training script
train-lens model.tlens      # opens a saved network
train-lens --help           # all options
```

Or from Python, in a script or notebook: `train_lens.launch()`. If a viewer
is already open, `launch()` does nothing.

## Training monitor

```python
mon = train_lens.monitor(model, optimizer)

for step, (x, y) in enumerate(loader):
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mon.step(step, loss=loss, acc=(out.argmax(1) == y).float().mean())  # before the next zero_grad

    if step % 500 == 0:
        mon.log(step, val_loss=evaluate(), val_acc=val_acc)  # sent as is, not averaged
```

| Panel | Shows | Read it like this |
|---|---|---|
| **Curves** | every metric you pass; `val_x` shares a chart with `x`; `lr` and total `grad_norm` are added for you | Training loss falling while validation loss rises means overfitting. Loss charts switch to a log scale when the range is wide. |
| **Layer health** | per layer: gradient norm (log bar) and update size \|ΔW\|/\|W\| per optimizer step | Healthy updates are around 1e-3. Above 5e-2 the LR is too high; below 1e-6 the layer barely learns. A gradient of 0 means nothing reaches the layer. |
| **Checks** | warnings with a suggested fix | see below |

| Check | Fires when |
|---|---|
| NaN / Inf loss | the latest loss is not finite |
| Loss spike | a loss point is more than 10× the first value |
| Diverging | recent loss is more than 3× the best seen |
| Plateau | the last quarter of the run did not beat the earlier best by 1% |
| Overfitting | `val_loss` is 5% above its best while training loss keeps falling |
| Exploding gradient | a layer's gradient norm is over 100 |
| No gradient | a trainable layer's gradient norm is about 0 (vanishing, or detached from the loss) |
| LR too high / too low | updates are over 5% / under 1e-6 of the weights per step |
| Dead ReLUs | over half of a hidden ReLU layer outputs 0 for every input (network view) |

Options: `monitor(model, optimizer, every=0.5, max_layers=40, track_updates=True, port=5757)`.
Metrics are averaged over `every` seconds. Large models are grouped into
`max_layers` rows. Update sizes need the optimizer and one temporary copy of
the trainable weights per send, and turn off automatically above 200M
trainable values. With LoRA, only the adapters count toward that limit.

## Network view

For MLPs, you can watch the neurons themselves, during training or inference:

```python
handle = train_lens.watch(model, labels=["cat", "dog", "bird"])
model(x)            # each forward pass lights up the network (max 20 per second)
handle.remove()
```

Connections glow by |activation × weight|. Gold means positive and red
means negative, and the signal travels from left to right. Hover a neuron to
see its value, and click it to mute it and watch how the output changes.
Layers wider than 64 neurons are drawn as an evenly spaced sample.

| Key / mouse | Action |
|---|---|
| drag / wheel | rotate / zoom |
| hover / click | show a neuron's value / mute it (`C` unmutes all) |
| `V` | flat or 3D view |
| `T` / `N` | show or hide the training panel / the network |
| `Space` | pause |
| `S` | save a PNG screenshot |
| `R` | randomize the weights |
| `[` `]` | connection brightness |
| `A` / `D` / `0` | auto-rotate / layer delay / reset camera |
| `H` / `Esc` | hide the key help / quit |

## Any framework

```python
v = train_lens.Viewer()
v.metric("loss", 0.42, step)                                  # curves
v.layer_stats([("encoder", grad_norm, weight_norm, update_ratio), ...])
v.topology([784, 128, 10], ["relu", "softmax"])               # network view
v.weights(0, W0, b0)                                          # shape (out, in)
v.input(x)                # the viewer runs the forward pass, or:
v.frame([x, h, y])        # send real activations
v.status(epoch=3)
```

Save a network to a file with `train_lens.save_torch(model, "net.tlens")` or
`train_lens.save_tlens(...)`. Open it later with `train-lens net.tlens`.

## How it works

```
 your training script                                 TrainLens window
┌─────────────────────────────┐                     ┌───────────────────────────┐
│ monitor() / watch() hooks   │  TCP 127.0.0.1:5757  │ C++ / OpenGL, 60+ fps     │
│ read params, grads, acts    │ ── plain text ──────▶│ curves · layer health     │
│ background thread, drops    │   one command/line   │ checks · network view     │
│ data rather than block      │                      │                           │
└─────────────────────────────┘                     └───────────────────────────┘
```

- **Separate process.** The viewer runs in its own process, so a slow frame
  never stalls a training step, and a crashed training run never takes the
  window down.
- **Lightweight hooks.** The Python side is a few hooks and a socket. It
  measures about every half second (`every=0.5`), so the cost per training
  step is close to zero.
- **Analysis lives in the viewer.** The checks run in the viewer, on the data
  it receives, so a client in any language gets them too.

## Protocol

`.tlens` files and the TCP stream (127.0.0.1:5757) use the same plain-text
commands, one per line. You can try them by hand with `nc localhost 5757`.

| Command | Meaning |
|---|---|
| `metric NAME VALUE [STEP]` | add a point to a curve (`nan` and `inf` are kept, so they can be flagged) |
| `pstats N` then `pstat I NAME GRAD WEIGHT UPDATE` | layer health table; -1 means unknown or no gradient |
| `clearmetrics` | drop all curves and layer stats |
| `layers 16 26 24` | layer sizes, input first |
| `act relu sigmoid` | activation per non-input layer: `relu tanh sigmoid softmax linear` |
| `w L v...` / `b L v...` | weights (row-major, `out x in`) / biases from layer L to L+1 |
| `input v...` | input; the viewer computes the other layers |
| `acts L v...` | real activations of layer L (0 = input) |
| `labels a b c` | output neuron names |
| `status text` | one line of text in the header |
| `reset` | forget sent activations and input |
| `# ...` | comment |

## Examples

```sh
git clone https://github.com/Yaiphaba189/TrainLens && cd TrainLens
# each example opens the viewer window itself with train_lens.launch()
python examples/torch_monitor_cnn.py          # CNN: the monitor flags overfitting
python examples/torch_monitor_cnn.py --lr 3   # learning rate far too high
python examples/torch_spiral_train.py         # MLP: network view plus monitor, then inference
python examples/numpy_xor_train.py            # no PyTorch: plain numpy
```

## Troubleshooting

| Problem | Fix |
|---|---|
| **Training on a remote GPU server over SSH** | The window needs a screen, so run it on your laptop and tunnel the port backwards: run `train-lens` locally, connect with `ssh -R 5757:localhost:5757 gpu-box`, and start training on the server as usual. |
| **Google Colab, Kaggle, or a headless server with no SSH from your machine** | Not supported yet: the viewer needs a display it can reach. |
| **`cannot find the train-lens-viewer program`** | The compiled viewer is missing. Reinstall with `pip install --force-reinstall train-lens`. |
| **`not listening for clients (port 5757 ...)`** | Another viewer is already open. Use that one, or pick another port: `train-lens --port 6000` and `monitor(..., port=6000)`. |
| **Curves appear but the layer table is empty, or every layer shows `NO GRAD`** | Call `mon.step()` after `loss.backward()` and before the next `optimizer.zero_grad()`. The monitor prints this hint once. |
| **No update sizes in the layer table** | Pass the optimizer: `monitor(model, optimizer)`. They also turn off above 200M trainable values. |
| **`pip install` compiles for a long time** | No prebuilt wheel exists for your platform, so it builds the viewer, and SDL2 if needed. This happens only once. |

## Limits

- The network view draws dense layers only: MLPs, or the MLP head of a
  bigger model. The training monitor has no such limit.
- `watch()` finds layers in the order they are registered, which matches
  `nn.Sequential`.
- The viewer accepts connections from this machine only (127.0.0.1). Use an
  SSH tunnel for remote training, as shown above.

## Roadmap

- **Windows support:** the viewer needs Winsock networking.
- **Remote and notebook training:** Colab and Kaggle.
- **Network view for convolution and attention layers.**
- **`monitor()` helpers for JAX and TensorFlow.**
- **Export a run** (curves and checks) to a report you can share.

Ideas and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed. Inspired by
[m607stars/Neural-Network-Visualization-using-OpenGL](https://github.com/m607stars/Neural-Network-Visualization-using-OpenGL).
