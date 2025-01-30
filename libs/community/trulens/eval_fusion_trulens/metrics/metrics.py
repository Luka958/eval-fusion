from types import MappingProxyType
from typing import Union

from eval_fusion_core.enums import MetricTag


class ContextRelevance:
    value = 'context_relevance'


class Groundedness:
    value = 'groundedness'


class Relevance:
    value = 'relevance'


TruLensMetric = Union[
    ContextRelevance,
    Groundedness,
    Relevance,
]


TAG_TO_METRIC_TYPES: dict[MetricTag, list[type[TruLensMetric]]] = MappingProxyType(
    {
        MetricTag.INPUT: [
            ContextRelevance,
            Relevance,
        ],
        MetricTag.OUTPUT: [
            ContextRelevance,
            Groundedness,
            Relevance,
        ],
        MetricTag.GROUND_TRUTH: [],
        MetricTag.RELEVANT_CHUNKS: [
            ContextRelevance,
            Groundedness,
        ],
        MetricTag.ALL: [
            ContextRelevance,
            Groundedness,
            Relevance,
        ],
    }
)
