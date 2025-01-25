from deepeval.models import DeepEvalBaseLLM
from eval_fusion_core.base import EvalFusionBaseLLM


class DeepEvalProxyLLM(DeepEvalBaseLLM):
    def __init__(self, llm_delegate: EvalFusionBaseLLM):
        self.llm_delegate = llm_delegate
        # super().__init__(model_name=llm_delegate.get_name())

    def load_model(self):
        return self.llm_delegate

    def generate(self, prompt: str) -> str:
        return self.llm_delegate.generate(prompt)

    async def a_generate(self, prompt: str) -> str:
        return await self.llm_delegate.a_generate(prompt)

    def get_model_name(self) -> str:
        return self.llm_delegate.get_name()
