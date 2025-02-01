from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseEM
from eval_fusion_core.models import TokenUsage
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class VertexAIEM(EvalFusionBaseEM):
    def __init__(self, model_name: str, output_dim: int):
        self._model_name = model_name
        self._output_dim = output_dim
        self._model = TextEmbeddingModel.from_pretrained(self._model_name)
        self.token_usage = TokenUsage()

    def get_name(self) -> str:
        self._model_name

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        task = 'RETRIEVAL_DOCUMENT'
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        kwargs = dict(output_dimensionality=768)
        embeddings = self._model.get_embeddings(inputs, **kwargs)

        count_tokens_response = self._model.count_tokens(texts)
        self.token_usage.add(count_tokens_response.total_tokens, 0)

        return [embedding.values for embedding in embeddings]

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.a_embed_texts([text])

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        task = 'RETRIEVAL_DOCUMENT'
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        kwargs = dict(output_dimensionality=768)
        embeddings = await self._model.get_embeddings_async(inputs, **kwargs)

        count_tokens_response = self._model.count_tokens(texts)
        self.token_usage.add(count_tokens_response.total_tokens, 0)

        return [embedding.values for embedding in embeddings]
