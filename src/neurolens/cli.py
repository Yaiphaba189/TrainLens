"""The ``neurolens`` command: runs the bundled viewer with the given options.

    neurolens                  # open the viewer and wait for training scripts
    neurolens model.nlens      # open a saved network
    neurolens --help           # all viewer options
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
        print(f"neurolens {__version__}")
        return 0
    try:
        exe = viewer_path()
    except FileNotFoundError as e:
        print(f"neurolens: {e}", file=sys.stderr)
        return 1
    if os.name == "posix" and from_command_line:
        os.execv(exe, [exe, *argv])  # replace this process, so Ctrl-C and exit codes behave
    return subprocess.call([exe, *argv])


if __name__ == "__main__":
    sys.exit(main())
