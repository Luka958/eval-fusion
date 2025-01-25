from typing import Any, Union

from eval_fusion_core.base import EvalFusionBaseLLM
from phoenix.evals.models import BaseModel
from phoenix.evals.templates import MultimodalPrompt


class PhoenixProxyLLM(BaseModel):
    def __init__(self, llm_delegate: EvalFusionBaseLLM):
        self.llm_delegate = llm_delegate

        super().__init__()  # TODO can base concurrency and rate limiter

    def _model_name(self) -> str:
        return self.llm_delegate.get_name()

    def _generate(self, prompt: Union[str, MultimodalPrompt], **kwargs: Any) -> str:
        if isinstance(prompt, MultimodalPrompt):
            prompt = prompt.to_text_only_prompt()

        return self.llm_delegate.generate(prompt)

    async def _async_generate(self, prompt: str, **kwargs: Any) -> str:
        if isinstance(prompt, MultimodalPrompt):
            prompt = prompt.to_text_only_prompt()

        return await self.llm_delegate.a_generate(prompt)
