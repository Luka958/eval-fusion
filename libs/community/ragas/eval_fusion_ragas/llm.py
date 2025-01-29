from typing import Optional

from eval_fusion_core.models.settings import EvalFusionLLMSettings
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.llms.base import BaseRagasLLM


class RagasProxyLLM(BaseRagasLLM):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt = prompt.to_string()
        result = self.__llm.generate(prompt)

        return LLMResult(generations=[[Generation(text=result)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt = prompt.to_string()
        result = await self.__llm.a_generate(prompt)

        return LLMResult(generations=[[Generation(text=result)]])
