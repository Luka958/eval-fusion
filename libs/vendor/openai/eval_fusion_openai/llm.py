from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseLLM
from eval_fusion_core.models import TokenUsage
from openai import NOT_GIVEN, AsyncOpenAI, OpenAI


class OpenAILLM(EvalFusionBaseLLM):
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        self._model_name = model_name
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.token_usage = TokenUsage()

    def get_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, use_json: bool = True) -> str:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'} if use_json else NOT_GIVEN,
        )
        self.token_usage.add(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )

        return completion.choices[0].message.content

    async def a_generate(self, prompt: str, use_json: bool = True) -> str:
        completion = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'} if use_json else NOT_GIVEN,
        )
        self.token_usage.add(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )

        return completion.choices[0].message.content

    def generate_from_messages(
        self, messages: list[dict], use_json: bool = True
    ) -> str:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            response_format={'type': 'json_object'} if use_json else NOT_GIVEN,
        )
        self.token_usage.add(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )

        return completion.choices[0].message.content

    def get_token_usage(self) -> TokenUsage:
        return self.token_usage
