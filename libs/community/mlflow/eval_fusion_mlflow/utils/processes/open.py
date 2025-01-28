from subprocess import PIPE, Popen


def open_process(args: list[str]) -> Popen[bytes]:
    return Popen(args, stdout=PIPE, stderr=PIPE)
