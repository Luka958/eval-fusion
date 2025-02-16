from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings


class RagCheckerProxyLLM:
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)

    def generate(self, prompt: str) -> str:  # TODO accepts multiple prompts
        return self.__llm.generate(prompt)

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
