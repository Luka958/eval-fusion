from types import MappingProxyType

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from eval_fusion_core.enums import MetricTag


TAGS_TO_METRICS: dict[MetricTag, list[type[BaseMetric]]] = MappingProxyType(
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
