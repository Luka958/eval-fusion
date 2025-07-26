from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings


class RagCheckerProxyLLM:
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)

    def custom_llm_api_func(self, prompts: list[str]) -> list[str]:
        return [self.__llm.generate(prompt, use_json=False) for prompt in prompts]

    async def a_custom_llm_api_func(self, prompts: list[str]) -> list[str]:
        return [
            await self.__llm.a_generate(prompt, use_json=False) for prompt in prompts
        ]

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
