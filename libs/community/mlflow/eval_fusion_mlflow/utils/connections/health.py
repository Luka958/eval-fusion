from time import sleep

from eval_fusion_core.exceptions import EvalFusionException
from requests import get
from requests.exceptions import ConnectionError


def check_health(host: str, port: int):
    try:
        for _ in range(5):
            try:
                models_serve_response = get(f'http://{host}:{port}/health')
                assert (
                    models_serve_response.status_code == 200,
                    f'Health check failed with status code {str(models_serve_response.status_code)}!',
                )

            except ConnectionError:
                sleep(2)

    except (ConnectionError, AssertionError) as e:
        raise EvalFusionException(str(e))
