from eval_fusion_core.models.settings import EvalFusionEMSettings
from ragas.embeddings.base import BaseRagasEmbeddings


class RagasProxyEM(BaseRagasEmbeddings):
    def __init__(self, settings: EvalFusionEMSettings):
        self.__em = settings.base_type(*settings.args, **settings.kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self.__em.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.__em.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.__em.a_embed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.__em.a_embed_texts(texts)
