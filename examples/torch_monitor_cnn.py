"""Monitors a small CNN while it trains: loss/accuracy curves for train and
validation, the learning rate, gradient flow per layer and automatic checks.

    python3 examples/torch_monitor_cnn.py             # healthy run, overfits near the end
    python3 examples/torch_monitor_cnn.py --lr 3      # see the viewer flag a learning rate that is too high

Nothing here is specific to CNNs: `train_lens.monitor(model, optimizer)` works for
any PyTorch model. The data is synthetic so the example needs no downloads.
"""

import argparse
import time

import torch
import torch.nn as nn

import train_lens

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--delay", type=float, default=0.03, help="seconds to pause per batch, so you can watch")
args = parser.parse_args()

torch.manual_seed(0)


def make_data(n):
    """16x16 images of 4 classes: a bar in one of four orientations, plus heavy noise."""
    y = torch.randint(0, 4, (n,))
    x = torch.randn(n, 1, 16, 16) * 1.6
    for i, c in enumerate(y.tolist()):
        if c == 0: x[i, 0, 7:9, :] += 0.9
        if c == 1: x[i, 0, :, 7:9] += 0.9
        if c == 2: x[i, 0].diagonal().add_(1.2)
        if c == 3: x[i, 0].flip(1).diagonal().add_(1.2)
    return x, y


x_train, y_train = make_data(320)  # small on purpose, so overfitting shows up
x_val, y_val = make_data(400)

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 4),
)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

train_lens.launch()  # opens the viewer window unless one is already running
mon = train_lens.monitor(model, opt)

step = 0
for epoch in range(args.epochs):
    model.train()
    perm = torch.randperm(len(x_train))
    for i in range(0, len(x_train), 32):
        idx = perm[i:i + 32]
        out = model(x_train[idx])
        loss = loss_fn(out, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        mon.step(step, loss=loss, acc=(out.argmax(1) == y_train[idx]).float().mean())
        time.sleep(args.delay)

    model.eval()
    with torch.no_grad():
        out = model(x_val)
        mon.log(step, val_loss=loss_fn(out, y_val), val_acc=(out.argmax(1) == y_val).float().mean())
    mon.status(f"epoch {epoch + 1}/{args.epochs}")

mon.viewer.close()
print("done")
