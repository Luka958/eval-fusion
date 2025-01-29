from decouple import config
from eval_fusion_core.base import EvalFusionBaseLLM
from openai import AsyncOpenAI, OpenAI


OPENAI_BASE_URL = config('OPENAI_BASE_URL')
OPENAI_API_KEY = config('OPENAI_API_KEY')


class OpenAILLM(EvalFusionBaseLLM):
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def get_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
        )

        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
        )

        return completion.choices[0].message.content

    def generate_from_messages(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={'type': 'json_object'},
        )

        return completion.choices[0].message.content
