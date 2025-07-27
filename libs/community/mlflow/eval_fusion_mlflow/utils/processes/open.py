import logging

from subprocess import PIPE, Popen
from sys import stderr, stdout
from threading import Thread


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _stream_logger(pipe, level: int):
    for raw in iter(pipe.readline, b''):
        line = raw.decode(errors='replace').rstrip()
        logger.log(level, line)


def open_process(args: list[str], log=False) -> Popen[bytes]:
    popen = Popen(
        args,
        stdout=PIPE if log else stdout,
        stderr=PIPE if log else stderr,
    )

    if log:
        stdout_thread = Thread(
            target=_stream_logger, args=(popen.stdout, logging.INFO), daemon=True
        )
        stderr_thread = Thread(
            target=_stream_logger, args=(popen.stderr, logging.ERROR), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

    return popen
