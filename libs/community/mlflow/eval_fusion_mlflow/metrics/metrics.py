from types import MappingProxyType
from typing import Any, Optional, Protocol

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
        model: Optional[str] = None,
        metric_version: Optional[str] = None,
        examples: Optional[list[EvaluationExample]] = None,
        metric_metadata: Optional[dict[str, Any]] = None,
        parameters: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        proxy_url: Optional[str] = None,
        max_workers: int = 10,
    ) -> EvaluationMetric:
        pass


MlFlowMetric = _MlFlowMetric


TAG_TO_METRIC_TYPES: dict[MetricTag, list[MlFlowMetric]] = MappingProxyType(
    {
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
)
