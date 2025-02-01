from __future__ import annotations

from subprocess import PIPE, Popen
from sys import stderr, stdout


def open_process(args: list[str], pipe_output=True) -> Popen[bytes]:
    return Popen(
        args,
        stdout=PIPE if pipe_output else stdout,
        stderr=PIPE if pipe_output else stderr,
    )
