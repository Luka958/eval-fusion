from eval_fusion_core.base import EvalFusionBaseEmbeddingModel
from langchain_core.callbacks import Callbacks
from ragas.embeddings.base import BaseRagasEmbeddings


class RagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedding_model_delegate: EvalFusionBaseEmbeddingModel):
        self.embedding_model_delegate = embedding_model_delegate

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_model_delegate.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model_delegate.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embedding_model_delegate.a_embed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedding_model_delegate.a_embed_texts(texts)
