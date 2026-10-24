"""The ``train-lens`` command: runs the bundled viewer with the given options.

    train-lens                  # open the viewer and wait for training scripts
    train-lens model.tlens      # open a saved network
    train-lens --help           # all viewer options
"""

import os
import subprocess
import sys

from . import __version__
from ._viewer import viewer_path


def main(argv=None):
    from_command_line = argv is None
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["-V"], ["--version"]):
        print(f"train-lens {__version__}")
        return 0
    try:
        exe = viewer_path()
    except FileNotFoundError as e:
        print(f"train-lens: {e}", file=sys.stderr)
        return 1
    if os.name == "posix" and from_command_line:
        os.execv(exe, [exe, *argv])  # replace this process, so Ctrl-C and exit codes behave
    return subprocess.call([exe, *argv])


if __name__ == "__main__":
    sys.exit(main())
