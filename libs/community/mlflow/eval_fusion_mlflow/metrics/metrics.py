from __future__ import annotations

from typing import Any, Protocol

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature
from mlflow.metrics.genai import (
    EvaluationExample,
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
)
from mlflow.models.evaluation import EvaluationMetric


class MlFlowMetric(EvalFusionBaseMetric):
    ANSWER_CORRECTNESS = 'answer_correctness'
    ANSWER_RELEVANCE = 'answer_relevance'
    ANSWER_SIMILARITY = 'answer_similarity'
    FAITHFULNESS = 'faithfulness'
    RELEVANCE = 'relevance'


class _MlFlowMetricType(Protocol):
    def __call__(
        self,
        model: str | None = None,
        metric_version: str | None = None,
        examples: list[EvaluationExample] | None = None,
        metric_metadata: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        proxy_url: str | None = None,
        max_workers: int = 10,
    ) -> EvaluationMetric:
        pass


MlFlowMetricType = _MlFlowMetricType

METRIC_TO_TYPE: dict[MlFlowMetric, MlFlowMetricType] = {
    MlFlowMetric.ANSWER_CORRECTNESS: answer_correctness,
    MlFlowMetric.ANSWER_RELEVANCE: answer_relevance,
    MlFlowMetric.ANSWER_SIMILARITY: answer_similarity,
    MlFlowMetric.FAITHFULNESS: faithfulness,
    MlFlowMetric.RELEVANCE: relevance,
}


FEATURE_TO_METRICS = {
    Feature.INPUT: [
        MlFlowMetric.ANSWER_CORRECTNESS,
        MlFlowMetric.ANSWER_RELEVANCE,
        MlFlowMetric.ANSWER_SIMILARITY,
        MlFlowMetric.FAITHFULNESS,
        MlFlowMetric.RELEVANCE,
    ],
    Feature.OUTPUT: [
        MlFlowMetric.ANSWER_CORRECTNESS,
        MlFlowMetric.ANSWER_RELEVANCE,
        MlFlowMetric.ANSWER_SIMILARITY,
        MlFlowMetric.FAITHFULNESS,
        MlFlowMetric.RELEVANCE,
    ],
    Feature.GROUND_TRUTH: [
        MlFlowMetric.ANSWER_CORRECTNESS,
        MlFlowMetric.ANSWER_SIMILARITY,
    ],
    Feature.RELEVANT_CHUNKS: [
        MlFlowMetric.FAITHFULNESS,
        MlFlowMetric.RELEVANCE,
    ],
    Feature.ALL: [
        MlFlowMetric.ANSWER_CORRECTNESS,
        MlFlowMetric.ANSWER_RELEVANCE,
        MlFlowMetric.ANSWER_SIMILARITY,
        MlFlowMetric.FAITHFULNESS,
        MlFlowMetric.RELEVANCE,
    ],
}
