"""Trains a 2-8-8-1 network on XOR with plain numpy while NeuroLens shows it.

Run it (it opens the viewer window itself):
    python3 examples/numpy_xor_train.py

Watch the connections reorganize as the loss falls. This example sends
weights plus an input and lets the viewer run the forward pass, which works
for any framework (JAX, TensorFlow, your own code).
"""

import argparse
import time

import numpy as np

import neurolens

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--delay", type=float, default=0.01, help="seconds to pause per update, so you can watch")
parser.add_argument("--port", type=int, default=5757)
args = parser.parse_args()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)
sizes = [2, 8, 8, 1]
acts = ["tanh", "tanh", "sigmoid"]

rng = np.random.default_rng(0)
W = [rng.normal(0, np.sqrt(1 / sizes[l]), (sizes[l + 1], sizes[l])) for l in range(3)]
b = [np.zeros(sizes[l + 1]) for l in range(3)]


def forward(x):
    a = [x]
    for l in range(3):
        z = a[-1] @ W[l].T + b[l]
        a.append(np.tanh(z) if l < 2 else 1 / (1 + np.exp(-z)))
    return a


neurolens.launch(port=args.port)  # opens the viewer window unless one is already running
viewer = neurolens.Viewer(port=args.port)
viewer.topology(sizes, acts, labels=["xor"])

lr = 0.5
for step in range(args.steps + 1):
    a = forward(X)
    out = a[-1]
    loss = float(np.mean((out - Y) ** 2))
    d = 2 * (out - Y) * out * (1 - out) / len(X)  # MSE through the sigmoid
    for l in range(2, -1, -1):
        gW, gb = d.T @ a[l], d.sum(0)
        if l:
            d = (d @ W[l]) * (1 - a[l] ** 2)  # through tanh, using W before the update
        W[l] -= lr * gW
        b[l] -= lr * gb

    if step % 10 == 0:
        for l in range(3):
            viewer.weights(l, W[l], b[l])
        case = int(time.monotonic()) % 4  # one XOR case per second, longer than the viewer's animation
        viewer.input(X[case])
        viewer.status(f"xor({int(X[case][0])},{int(X[case][1])})", step=step, loss=loss)
        time.sleep(args.delay)

print("predictions:", np.round(forward(X)[-1].ravel(), 3), "targets:", Y.ravel())
viewer.close()
