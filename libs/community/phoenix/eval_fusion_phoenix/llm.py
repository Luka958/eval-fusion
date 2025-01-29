from typing import Any, Union

from eval_fusion_core.models.settings import EvalFusionLLMSettings
from phoenix.evals.models import BaseModel
from phoenix.evals.templates import MultimodalPrompt


class PhoenixProxyLLM(BaseModel):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.__llm = settings.base_type(*settings.args, **settings.kwargs)
        super().__init__(default_concurrency=1)

    def _model_name(self) -> str:
        return self.__llm.get_name()

    def _generate(self, prompt: Union[str, MultimodalPrompt], **kwargs: Any) -> str:
        if isinstance(prompt, MultimodalPrompt):
            prompt = prompt.to_text_only_prompt()

        return self.__llm.generate(prompt)

    async def _async_generate(self, prompt: str, **kwargs: Any) -> str:
        if isinstance(prompt, MultimodalPrompt):
            prompt = prompt.to_text_only_prompt()

        return await self.__llm.a_generate(prompt)
