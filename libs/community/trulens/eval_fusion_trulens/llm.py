from typing import Optional, Sequence

from eval_fusion_core.models.settings import EvalFusionLLMSettings
from trulens.feedback import LLMProvider


class TruLensProxyLLM(LLMProvider):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)
        super().__init__()

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[dict]] = None,
        **kwargs,
    ) -> str:
        if prompt:
            return self.__llm.generate(prompt)

        return self.__llm.generate_from_messages(messages)
