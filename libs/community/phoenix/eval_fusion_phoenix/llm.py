from __future__ import annotations

from typing import Any

from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from phoenix.evals.models import BaseModel
from phoenix.evals.templates import MultimodalPrompt


class PhoenixProxyLLM(BaseModel):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)
        super().__init__(default_concurrency=1)

    def _model_name(self) -> str:
        return self.__llm.get_name()

    def _generate(self, prompt: str | MultimodalPrompt, **kwargs: Any) -> str:
        if isinstance(prompt, MultimodalPrompt):
            prompt = prompt.to_text_only_prompt()

        return self.__llm.generate(prompt, use_json=False)

    async def _async_generate(self, prompt: str, **kwargs: Any) -> str:
        if isinstance(prompt, MultimodalPrompt):
            prompt = prompt.to_text_only_prompt()

        return await self.__llm.a_generate(prompt, use_json=False)

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
