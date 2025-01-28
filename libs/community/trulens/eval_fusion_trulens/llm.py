from typing import Optional, Sequence

from eval_fusion_core.base import EvalFusionBaseLLM
from trulens.feedback import LLMProvider


class TruLensProxyLLM(LLMProvider):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.__llm = llm

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[dict]] = None,
        **kwargs,
    ) -> str:
        if prompt:
            return self.__llm.generate(prompt)

        return self.__llm.generate_from_messages(messages)
