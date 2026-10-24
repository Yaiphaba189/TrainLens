"""Finds and starts the compiled NeuroLens viewer that ships with the package."""

import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import time

__all__ = ["viewer_path", "launch"]

EXE = "neurolens-viewer"


def viewer_path():
    """Path of the viewer executable.

    Looks at $NEUROLENS_VIEWER, then the scripts folder of the current Python
    environment (where pip installs it), then PATH.
    """
    candidates = []
    if os.environ.get("NEUROLENS_VIEWER"):
        candidates.append(os.environ["NEUROLENS_VIEWER"])
    for scheme_path in (sysconfig.get_path("scripts"), os.path.dirname(sys.executable)):
        if scheme_path:
            candidates.append(os.path.join(scheme_path, EXE))
    found = shutil.which(EXE)
    if found:
        candidates.append(found)
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        f"cannot find the {EXE} program. It is built when neurolens is installed with pip; "
        "reinstall with `pip install --force-reinstall neurolens`, or point $NEUROLENS_VIEWER at a "
        "viewer you built with CMake (see CONTRIBUTING.md)."
    )


def _listening(port, timeout=0.2):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def launch(*args, port=5757, wait=10.0):
    """Opens the viewer window in its own process, so a script or notebook
    can start it without a second terminal.

    Returns the ``subprocess.Popen`` of the new window, or ``None`` when a
    viewer is already listening on ``port``. Extra ``args`` are passed to the
    viewer, e.g. ``launch("--3d")``.
    """
    if _listening(port):
        return None
    proc = subprocess.Popen([viewer_path(), "--port", str(port), *map(str, args)])
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the viewer exited with code {proc.returncode}; run `neurolens` in a terminal to see why")
        if _listening(port):
            return proc
        time.sleep(0.1)
    return proc  # still starting; clients retry on their own
