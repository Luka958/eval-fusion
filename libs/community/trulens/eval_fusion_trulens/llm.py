from typing import Optional, Sequence

from eval_fusion_core.base import EvalFusionBaseLLM
from trulens.feedback import LLMProvider


class TrulensLLM(LLMProvider):
    def __init__(self, llm_delegate: EvalFusionBaseLLM):
        self.llm_delegate = llm_delegate

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[dict]] = None,
        **kwargs,
    ) -> str:
        if prompt:
            return self.llm_delegate.generate(prompt)

        return self.llm_delegate.generate_from_messages(messages)
