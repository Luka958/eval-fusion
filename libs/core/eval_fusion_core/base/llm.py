from abc import ABC, abstractmethod

from eval_fusion_core.models import TokenUsage


class EvalFusionBaseLLM(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, use_json: bool) -> str:
        pass

    @abstractmethod
    async def a_generate(self, prompt: str, use_json: bool) -> str:
        pass

    @abstractmethod
    def generate_from_messages(self, messages: list[dict], use_json: bool) -> str:
        pass

    @abstractmethod
    def get_token_usage(self) -> TokenUsage:
        pass
