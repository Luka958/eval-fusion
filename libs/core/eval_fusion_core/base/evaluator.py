from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from eval_fusion_core.enums import Feature
from eval_fusion_core.models import EvaluationInput, EvaluationOutput

from .metric import EvalFusionBaseMetric


class EvalFusionBaseEvaluator(ABC):
    @abstractmethod
    def __enter__(self) -> EvalFusionBaseEvaluator:
        pass

    @abstractmethod
    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[EvalFusionBaseMetric] | None,
        feature: Feature | None,
    ) -> list[EvaluationOutput]:
        pass

    @abstractmethod
    async def a_evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[EvalFusionBaseMetric] | None,
        feature: Feature | None,
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
