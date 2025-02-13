from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import EvaluationInput, EvaluationOutput


class EvalFusionBaseEvaluator(ABC):
    @abstractmethod
    def __enter__(self) -> EvalFusionBaseEvaluator:
        pass

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type],
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
