from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from eval_fusion_core.models import EvaluationInput, EvaluationOutput

from .embedding_model import EvalFusionBaseEmbeddingModel
from .llm import EvalFusionBaseLLM


class EvalFusionBaseEvaluator(ABC):
    embedding_model: EvalFusionBaseEmbeddingModel
    llm: EvalFusionBaseLLM

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        pass

    @abstractmethod
    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        pass
