from __future__ import annotations

from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig


class RagasProxyLLM(BaseRagasLLM):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)
        self.run_config = RunConfig(
            timeout=60,
            max_retries=0,
            max_wait=0,
            max_workers=1,
            exception_types=(),
            log_tenacity=False,
            seed=42,
        )

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt = prompt.to_string()
        result = self.__llm.generate(prompt)

        return LLMResult(generations=[[Generation(text=result)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt = prompt.to_string()
        result = await self.__llm.a_generate(prompt)

        return LLMResult(generations=[[Generation(text=result)]])

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
