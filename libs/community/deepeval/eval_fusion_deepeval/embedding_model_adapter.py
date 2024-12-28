from deepeval.models import DeepEvalBaseEmbeddingModel


class EmbeddingModelAdapter(DeepEvalBaseEmbeddingModel):
    def load_model(self):
        return ...  # TODO return model object

    def embed_text(self, text: str) -> list[float]:
        embedding_model = self.load_model()
        result = embedding_model.embed_query(text)

        return ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedding_model = self.load_model()
        results = embedding_model.embed_documents(texts)

        return ...

    async def a_embed_text(self, text: str) -> list[float]:
        embedding_model = self.load_model()

        return await ...

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedding_model = self.load_model()

        return await ...

    def get_model_name(self):
        return ...
