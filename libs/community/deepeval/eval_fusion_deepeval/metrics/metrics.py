from types import MappingProxyType
from typing import Type, Union

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from eval_fusion_core.enums import MetricTag


DeepEvalMetric = Union[
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
]

TAG_TO_METRICS: dict[MetricTag, list[type[DeepEvalMetric]]] = MappingProxyType(
    {
        MetricTag.INPUT: [
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        ],
        MetricTag.OUTPUT: [
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        ],
        MetricTag.GROUND_TRUTH: [
            ContextualPrecisionMetric,
            ContextualRecallMetric,
        ],
        MetricTag.RELEVANT_CHUNKS: [
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        ],
        MetricTag.ALL: [
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        ],
    }
)
