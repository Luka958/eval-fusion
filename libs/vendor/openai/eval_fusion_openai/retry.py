import backoff

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)


def with_backoff():
    return backoff.on_exception(
        backoff.expo,
        exception=(
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_tries=10,
    )
