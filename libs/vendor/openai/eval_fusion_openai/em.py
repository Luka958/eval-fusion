from eval_fusion_core.base import EvalFusionBaseEM
from eval_fusion_core.models import TokenUsage
from openai import AsyncOpenAI, OpenAI


class OpenAIEM(EvalFusionBaseEM):
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

    def embed_text(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text, model=self._model_name, encoding_format='float'
        )
        self.token_usage.add(response.usage.prompt_tokens, 0)

        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts, model=self._model_name, encoding_format='float'
        )
        self.token_usage.add(response.usage.prompt_tokens, 0)

        return [x.embedding for x in response.data]

    async def a_embed_text(self, text: str) -> list[float]:
        response = await self._async_client.embeddings.create(
            input=text, model=self._model_name, encoding_format='float'
        )
        self.token_usage.add(response.usage.prompt_tokens, 0)

        return response.data[0].embedding

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self._async_client.embeddings.create(
            input=texts, model=self._model_name, encoding_format='float'
        )
        self.token_usage.add(response.usage.prompt_tokens, 0)

        return [x.embedding for x in response.data]

    def get_token_usage(self) -> TokenUsage:
        return self.token_usage
