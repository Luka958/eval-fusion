from psutil import Process  # type: ignore


def close_process(pid: int):
    parent = Process(pid)

    for child in parent.children(recursive=True):
        child.terminate()

    parent.terminate()
