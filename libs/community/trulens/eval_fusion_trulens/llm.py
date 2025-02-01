from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseLLM
from eval_fusion_core.models import TokenUsage
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
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> str:
        if prompt:
            return self.__llm.generate(prompt)

        return self.__llm.generate_from_messages(messages, use_json=False)

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
