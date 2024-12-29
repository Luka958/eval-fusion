from abc import ABC, abstractmethod


class EvalFusionBaseLLM(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def a_generate(self, prompt: str) -> str:
        pass
