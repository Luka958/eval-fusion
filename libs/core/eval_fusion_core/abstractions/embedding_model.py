from abc import ABC, abstractmethod


class EvalFusionBaseEmbeddingModel(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    async def a_embed_text(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass
