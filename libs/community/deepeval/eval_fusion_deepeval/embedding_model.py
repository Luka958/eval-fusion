from deepeval.models import DeepEvalBaseEmbeddingModel
from eval_fusion_core.abstractions import EvalFusionBaseEmbeddingModel


class DeepEvalEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, embedding_model_delegate: EvalFusionBaseEmbeddingModel):
        self.embedding_model_delegate = embedding_model_delegate
        # super().__init__(model_name=embedding_model_delegate.get_name())

    def load_model(self):
        return self.embedding_model_delegate

    def embed_text(self, text: str) -> list[float]:
        return self.embedding_model_delegate.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model_delegate.embed_texts(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.embedding_model_delegate.a_embed_text(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.embedding_model_delegate.a_embed_texts(texts)

    def get_model_name(self) -> str:
        return self.embedding_model_delegate.get_name()
