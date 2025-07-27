import backoff

from eval_fusion_core.exceptions import EvalFusionException
from requests import get
from requests.exceptions import ConnectionError


@backoff.on_exception(backoff.constant, ConnectionError, max_tries=5, interval=5)
def _check_health(host: str, port: int):
    response = get(f'http://{host}:{port}/health')
    response.raise_for_status()


def check_health(host: str, port: int):
    try:
        _check_health(host, port)

    except ConnectionError as e:
        raise EvalFusionException(str(e))
