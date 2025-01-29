from enum import Enum
from types import MappingProxyType

from eval_fusion_core.enums import MetricTag


class _TruLensEnumMetric(str, Enum):
    CONTEXT_RELEVANCE = 'context_relevance'
    GROUNDEDNESS = 'groundedness'
    ANSWER_RELEVANCE = 'answer_relevance'


TruLensMetric = _TruLensEnumMetric


TAG_TO_METRIC_TYPES: dict[MetricTag, list[type[TruLensMetric]]] = MappingProxyType(
    {
        MetricTag.INPUT: [
            TruLensMetric.CONTEXT_RELEVANCE,
            TruLensMetric.ANSWER_RELEVANCE,
        ],
        MetricTag.OUTPUT: [
            TruLensMetric.CONTEXT_RELEVANCE,
            TruLensMetric.GROUNDEDNESS,
            TruLensMetric.ANSWER_RELEVANCE,
        ],
        MetricTag.GROUND_TRUTH: [],
        MetricTag.RELEVANT_CHUNKS: [
            TruLensMetric.CONTEXT_RELEVANCE,
            TruLensMetric.GROUNDEDNESS,
        ],
        MetricTag.ALL: [
            TruLensMetric.CONTEXT_RELEVANCE,
            TruLensMetric.GROUNDEDNESS,
            TruLensMetric.ANSWER_RELEVANCE,
        ],
    }
)
