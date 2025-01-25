from eval_fusion_core.base import EvalFusionBaseEmbeddingModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class VertexAIEmbeddingModel(EvalFusionBaseEmbeddingModel):
    def __init__(self, model_name: str, output_dimensionality: int):
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality

        self.model = TextEmbeddingModel.from_pretrained(self.model_name)

    def get_name(self) -> str:
        self.model_name

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        task = 'RETRIEVAL_DOCUMENT'
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        kwargs = dict(output_dimensionality=768)
        embeddings = self.model.get_embeddings(inputs, **kwargs)

        return [embedding.values for embedding in embeddings]

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.a_embed_texts([text])

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        task = 'RETRIEVAL_DOCUMENT'
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        kwargs = dict(output_dimensionality=768)
        embeddings = await self.model.get_embeddings_async(inputs, **kwargs)

        return [embedding.values for embedding in embeddings]
