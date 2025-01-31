from subprocess import PIPE, Popen
from sys import stderr, stdout


def open_process(args: list[str]) -> Popen[bytes]:
    return Popen(
        args,
        stdout=stdout if args[1] == 'models' else PIPE,
        stderr=stderr if args[1] == 'models' else PIPE,
    )
