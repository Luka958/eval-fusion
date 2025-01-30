from typing import Optional, Sequence

from eval_fusion_core.base import EvalFusionBaseLLM
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from pydantic import PrivateAttr
from trulens.feedback import LLMProvider


class TruLensProxyLLM(LLMProvider):
    settings: EvalFusionLLMSettings
    __llm: EvalFusionBaseLLM = PrivateAttr()

    def model_post_init(self, __context: dict[str, object] | None = None) -> None:
        self.__llm = self.settings.base_type(
            *self.settings.args, **self.settings.kwargs
        )

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[dict]] = None,
        **kwargs,
    ) -> str:
        if prompt:
            return self.__llm.generate(prompt)

        return self.__llm.generate_from_messages(messages, use_json=False)
