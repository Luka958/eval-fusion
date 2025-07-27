import backoff
import httpx
import requests

from eval_fusion_core.exceptions import EvalFusionException


@backoff.on_exception(backoff.constant, httpx.ConnectError, max_tries=5, interval=5)
def _check_health(host: str, port: int) -> None:
    response = httpx.get(f'http://{host}:{port}/health')
    response.raise_for_status()


def check_health(host: str, port: int) -> None:
    try:
        _check_health(host, port)
    except httpx.ConnectError as e:
        raise EvalFusionException(str(e))


@backoff.on_exception(backoff.constant, httpx.ConnectError, max_tries=5, interval=5)
async def _a_check_health(host: str, port: int) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f'http://{host}:{port}/health')
        resp.raise_for_status()


async def a_check_health(host: str, port: int) -> None:
    try:
        await _a_check_health(host, port)
    except httpx.ConnectError as e:
        raise EvalFusionException(str(e))
