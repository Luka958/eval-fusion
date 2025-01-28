from deepeval.models import DeepEvalBaseLLM
from eval_fusion_core.base import EvalFusionBaseLLM


class DeepEvalProxyLLM(DeepEvalBaseLLM):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.__llm = llm
        # super().__init__(model_name=llm_delegate.get_name())

    def load_model(self):
        return self.__llm

    def generate(self, prompt: str) -> str:
        return self.__llm.generate(prompt)

    async def a_generate(self, prompt: str) -> str:
        return await self.__llm.a_generate(prompt)

    def get_model_name(self) -> str:
        return self.__llm.get_name()
