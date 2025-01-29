from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import EvaluationInput, EvaluationOutput


class EvalFusionBaseEvaluator(ABC):
    @abstractmethod
    def __enter__(self) -> Self:
        pass

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type],
        tag: MetricTag | None,
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
