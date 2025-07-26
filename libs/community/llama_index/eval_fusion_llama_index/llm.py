from __future__ import annotations

from typing import Any, AsyncGenerator, Sequence

from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

# NOTE: evaluation requires LLM instead of BaseLLM
# from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms.llm import LLM


class LlamaIndexProxyLLM(LLM):
    def __init__(self, settings: EvalFusionLLMSettings):
        super().__init__()
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.__llm.get_name())

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages = [{message.role: message.content} for message in messages]
        result = self.__llm.generate_from_messages(messages, use_json=False)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=result)
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        result = self.__llm.generate(prompt, use_json=False)

        return CompletionResponse(text=result)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError()

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        messages = [{message.role: message.content} for message in messages]
        result = await self.__llm.a_generate_from_messages(messages, use_json=False)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=result)
        )

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        result = await self.__llm.a_generate(prompt, use_json=False)

        return CompletionResponse(text=result)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        raise NotImplementedError()

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
