from __future__ import annotations

from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings


class TonicValidateProxyLLM(...):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
