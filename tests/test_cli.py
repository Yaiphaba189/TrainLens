"""Checks the installed `train-lens` command and the viewer it runs (no window needed)."""

import shutil
import subprocess
from pathlib import Path

import pytest

import train_lens
from train_lens.cli import main

ROOT = Path(__file__).resolve().parents[1]

try:
    VIEWER = train_lens.viewer_path()
except FileNotFoundError:
    VIEWER = None

needs_viewer = pytest.mark.skipif(VIEWER is None, reason="viewer not built (pip install the package first)")


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"train-lens {train_lens.__version__}"


@needs_viewer
def test_viewer_selftest():
    out = subprocess.run([VIEWER, "--selftest"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "all checks passed" in out.stdout


@needs_viewer
def test_cli_passes_options_to_viewer():
    assert main(["--check", str(ROOT / "examples" / "xor.tlens")]) == 0


@needs_viewer
def test_saved_file_opens_in_viewer(tmp_path):
    path = tmp_path / "net.tlens"
    train_lens.save_tlens(str(path), [3, 2], ["softmax"], weights=[[[1, 0, 0], [0, 1, 0]]], labels=["a", "b"])
    out = subprocess.run([VIEWER, "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "3 -> 2" in out.stdout


@needs_viewer
def test_bad_file_is_reported(tmp_path):
    path = tmp_path / "bad.tlens"
    path.write_text("layers 2 2\nw 0 1 2 3\n")
    out = subprocess.run([VIEWER, "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 1
    assert "bad.tlens:2" in out.stderr


def test_missing_viewer_message(monkeypatch, capsys):
    monkeypatch.setenv("TRAIN_LENS_VIEWER", "/nonexistent")
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    monkeypatch.setattr(train_lens._viewer.sysconfig, "get_path", lambda *_: "/nonexistent")
    monkeypatch.setattr(train_lens._viewer.sys, "executable", "/nonexistent/python")
    assert main([]) == 1
    assert "cannot find the train-lens-viewer program" in capsys.readouterr().err
