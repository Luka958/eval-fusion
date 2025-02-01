from __future__ import annotations

from typing import Any, Protocol

from eval_fusion_core.enums import MetricTag
from mlflow.metrics.genai import (
    EvaluationExample,
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
)
from mlflow.models.evaluation import EvaluationMetric


class _MlFlowMetric(Protocol):
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


MlFlowMetric = _MlFlowMetric


TAG_TO_METRIC_TYPES: dict[MetricTag, list[MlFlowMetric]] = {
    MetricTag.INPUT: [
        answer_correctness,
        answer_relevance,
        answer_similarity,
        faithfulness,
        relevance,
    ],
    MetricTag.OUTPUT: [
        answer_correctness,
        answer_relevance,
        answer_similarity,
        faithfulness,
        relevance,
    ],
    MetricTag.GROUND_TRUTH: [
        answer_correctness,
        answer_similarity,
    ],
    MetricTag.RELEVANT_CHUNKS: [
        faithfulness,
        relevance,
    ],
    MetricTag.ALL: [
        answer_correctness,
        answer_relevance,
        answer_similarity,
        faithfulness,
        relevance,
    ],
}
