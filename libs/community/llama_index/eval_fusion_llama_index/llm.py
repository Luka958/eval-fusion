from __future__ import annotations

from typing import Any, AsyncGenerator, Sequence

from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)


class LlamaIndexProxyLLM(BaseLLM):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        pass

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        pass

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        pass

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        pass

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        pass

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        pass

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        pass

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
