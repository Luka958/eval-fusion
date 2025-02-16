from __future__ import annotations

from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionEMSettings
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding


class LlamaIndexProxyEM(BaseEmbedding):
    def __init__(self, settings: EvalFusionEMSettings):
        self.__em = settings.base_type(*settings.args, **settings.kwargs)

    def _get_query_embedding(self, query: str) -> Embedding:
        pass

    async def _aget_query_embedding(self, query: str) -> Embedding:
        pass

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_query_embedding(text)

    def get_token_usage(self) -> TokenUsage:
        return self.__em.get_token_usage()
