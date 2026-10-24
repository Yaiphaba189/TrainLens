import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path


import train_lens


class FakeViewer:
    """Accepts one client and records the lines it sends."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.lines = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        conn, _ = self.srv.accept()
        self.srv.close()
        buf = b""
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                buf += data
                *done, buf = buf.split(b"\n")
                self.lines += [l.decode() for l in done]

    def wait_for(self, prefix, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for line in list(self.lines):
                if line.startswith(prefix):
                    return line
            time.sleep(0.01)
        raise AssertionError(f"no line starting with {prefix!r}; got {self.lines}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ViewerTest(unittest.TestCase):
    def test_sends_topology_weights_frames_and_status(self):
        fake = FakeViewer()
        v = train_lens.Viewer(port=fake.port, quiet=True)
        v.topology([2, 3, 1], ["relu", "sigmoid"], labels=["is yes"])
        v.weights(0, [[1, 2], [3, 4], [5, 6]], [0.1, 0.2, 0.3])
        v.frame([[0.5, -0.5], [1, 0, 2], [0.9]])
        v.status(step=3, loss=0.25)
        self.assertEqual(fake.wait_for("layers "), "layers 2 3 1")
        self.assertEqual(fake.wait_for("act "), "act relu sigmoid")
        self.assertEqual(fake.wait_for("labels "), "labels is_yes")
        self.assertEqual(fake.wait_for("w 0 "), "w 0 1 2 3 4 5 6")
        self.assertEqual(fake.wait_for("b 0 "), "b 0 0.1 0.2 0.3")
        self.assertEqual(fake.wait_for("acts 1 "), "acts 1 1 0 2")
        self.assertEqual(fake.wait_for("status "), "status step=3  loss=0.25")
        # the header always reaches the viewer before data that depends on it
        self.assertLess(fake.lines.index("layers 2 3 1"), fake.lines.index("acts 1 1 0 2"))
        v.close()

    def test_numpy_arrays(self):
        import numpy as np

        fake = FakeViewer()
        v = train_lens.Viewer(port=fake.port, quiet=True)
        v.topology([2, 2], ["linear"])
        v.weights(0, np.eye(2, dtype=np.float32))
        v.input(np.array([0.25, 1.5]))
        self.assertEqual(fake.wait_for("w 0 "), "w 0 1 0 0 1")
        self.assertEqual(fake.wait_for("input "), "input 0.25 1.5")
        v.close()

    def test_no_viewer_never_blocks(self):
        v = train_lens.Viewer(port=free_port(), quiet=True)
        t0 = time.monotonic()
        for _ in range(5000):
            v.acts(0, [0.0, 1.0])
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertFalse(v.connected)
        v.close()

    def test_bad_topology_rejected(self):
        v = train_lens.Viewer(port=free_port(), quiet=True)
        with self.assertRaises(ValueError):
            v.topology([2, 3, 1], ["relu"])
        v.close()


class MetricsTest(unittest.TestCase):
    def test_curves_logged_before_the_viewer_starts_are_replayed(self):
        port = free_port()
        v = train_lens.Viewer(port=port, quiet=True, retry_every=0.1)
        for step in range(5):
            v.metrics(step, loss=1.0 / (step + 1))
        v.metric("val_loss", float("nan"), 4)
        time.sleep(0.2)  # the client has failed to connect at least once
        fake = FakeViewer.__new__(FakeViewer)
        fake.srv = socket.socket()
        fake.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        fake.srv.bind(("127.0.0.1", port))
        fake.srv.listen(1)
        fake.lines = []
        threading.Thread(target=fake._run, daemon=True).start()
        fake.wait_for("metric val_loss")
        metric_lines = [l for l in fake.lines if l.startswith(("metric", "clearmetrics"))]
        self.assertEqual(metric_lines[0], "clearmetrics")
        self.assertEqual(metric_lines[1], "metric loss 1 0")
        self.assertEqual(metric_lines[-1], "metric val_loss nan 4")
        self.assertEqual(len(metric_lines), 7)
        v.close()

    def test_layer_stats(self):
        fake = FakeViewer()
        v = train_lens.Viewer(port=fake.port, quiet=True)
        v.layer_stats([("enc.fc1", 0.5, 3.0, 0.001), ("head", -1, 2.0, -1)])
        self.assertEqual(fake.wait_for("pstats"), "pstats 2")
        self.assertEqual(fake.wait_for("pstat 0"), "pstat 0 enc.fc1 0.5 3 0.001")
        self.assertEqual(fake.wait_for("pstat 1"), "pstat 1 head -1 2 -1")
        v.close()


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchTest(unittest.TestCase):
    def test_monitor_any_model(self):
        import torch.nn as nn

        fake = FakeViewer()
        model = nn.Sequential(nn.Conv1d(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6, 2))
        model[0].requires_grad_(False)  # a frozen layer is left out of the table
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        mon = train_lens.monitor(model, opt, viewer=train_lens.Viewer(port=fake.port, quiet=True), every=0)
        x, y = torch.randn(8, 1, 8), torch.randint(0, 2, (8,))
        for _ in range(3):
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mon.step(loss=loss)
        mon.log(val_loss=0.5)
        self.assertTrue(fake.wait_for("metric loss ").endswith(" 1"))
        self.assertTrue(fake.wait_for("metric lr ").startswith("metric lr 0.1"))
        self.assertTrue(fake.wait_for("metric grad_norm "))
        self.assertEqual(fake.wait_for("metric val_loss"), "metric val_loss 0.5 3")
        self.assertEqual(fake.wait_for("pstats"), "pstats 1")
        row = fake.wait_for("pstat 0 ").split()
        self.assertEqual(row[2], "3:Linear")
        self.assertGreater(float(row[3]), 0)                  # gradient norm
        time.sleep(0.3)
        rows = [l.split() for l in fake.lines if l.startswith("pstat 0 ")]
        self.assertTrue(any(0 < float(r[5]) < 1 for r in rows))  # update ratio measured after a step
        mon.remove()
        mon.viewer.close()

    def test_watch_mlp(self):
        import torch.nn as nn

        fake = FakeViewer()
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Softmax(dim=1))
        h = train_lens.watch(model, viewer=train_lens.Viewer(port=fake.port, quiet=True), max_hz=0)
        model(torch.randn(5, 3))
        self.assertEqual(fake.wait_for("layers "), "layers 3 4 2")
        self.assertEqual(fake.wait_for("act "), "act relu softmax")
        out = [float(v) for v in fake.wait_for("acts 2 ").split()[2:]]
        self.assertAlmostEqual(sum(out), 1.0, places=4)       # softmax applied to the shown sample
        h.remove()
        h.viewer.close()


class SaveNitTest(unittest.TestCase):
    def test_writes_commands(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.tlens")
            train_lens.save_tlens(path, [2, 1], ["sigmoid"], weights=[[[0.5, -1]]], biases=[[2]],
                              labels=["out"], input=[1, 0])
            lines = Path(path).read_text().splitlines()
        self.assertIn("layers 2 1", lines)
        self.assertIn("act sigmoid", lines)
        self.assertIn("w 0 0.5 -1", lines)
        self.assertIn("b 0 2", lines)
        self.assertIn("input 1 0", lines)


if __name__ == "__main__":
    unittest.main()
