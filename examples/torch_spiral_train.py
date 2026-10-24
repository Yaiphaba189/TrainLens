"""Trains a PyTorch MLP on a 3-class spiral, then runs inference, while
TrainLens shows the real activations from each forward pass.

    python3 examples/torch_spiral_train.py        # needs: pip install torch

The only TrainLens lines are `train_lens.watch(model, ...)` and `handle.status(...)`.
"""

import math
import time

import torch
import torch.nn as nn

import train_lens

torch.manual_seed(0)


def spiral(n_per_class=200, classes=3):
    xs, ys = [], []
    for c in range(classes):
        r = torch.linspace(0.05, 1, n_per_class)
        t = r * 4 + c * 2 * math.pi / classes + torch.randn(n_per_class) * 0.15
        xs.append(torch.stack([r * torch.sin(t), r * torch.cos(t)], 1))
        ys.append(torch.full((n_per_class,), c))
    return torch.cat(xs), torch.cat(ys)


X, y = spiral()
model = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 3), nn.Softmax(dim=1),
)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

train_lens.launch()  # opens the viewer window unless one is already running
handle = train_lens.watch(model, labels=["red", "green", "blue"], max_hz=20, weights_every=0.5)
mon = train_lens.monitor(model, opt, viewer=handle.viewer)  # curves and layer health beside the network

# ---- training ----
for epoch in range(300):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        idx = perm[i:i + 32]
        probs = model(X[idx])
        loss = nn.functional.nll_loss(torch.log(probs + 1e-9), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        mon.step(loss=loss)
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item()
    mon.log(acc=acc)
    handle.status("training", epoch=epoch, loss=loss.item(), acc=acc)
    time.sleep(0.02)

# ---- inference: one point at a time, slow enough to follow ----
model.eval()
with torch.no_grad():
    for k in range(60):
        i = (k * 37) % len(X)
        probs = model(X[i:i + 1])
        pred = probs.argmax(1).item()
        handle.status("inference", point=i, truth=["red", "green", "blue"][y[i]], pred=["red", "green", "blue"][pred])
        time.sleep(0.5)

handle.remove()
mon.remove()
handle.viewer.close()
print(f"final accuracy {acc:.3f}")
