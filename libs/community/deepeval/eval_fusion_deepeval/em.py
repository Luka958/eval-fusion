from __future__ import annotations

from deepeval.models import DeepEvalBaseEmbeddingModel
from eval_fusion_core.base import EvalFusionBaseEM


class DeepEvalProxyEM(DeepEvalBaseEmbeddingModel):
    def __init__(self, em: EvalFusionBaseEM):
        self.__em = em
        # super().__init__(model_name=embedding_model_delegate.get_name())

    def load_model(self):
        return self.__em

    def embed_text(self, text: str) -> list[float]:
        return self.__em.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.__em.embed_texts(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.__em.a_embed_text(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.__em.a_embed_texts(texts)

    def get_model_name(self) -> str:
        return self.__em.get_name()
