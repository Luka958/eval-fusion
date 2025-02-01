from __future__ import annotations

from eval_fusion_core.enums import MetricTag


class ContextRelevance:
    pass


class Groundedness:
    pass


class Relevance:
    pass


TruLensMetric = ContextRelevance | Groundedness | Relevance


TAG_TO_METRIC_TYPES: dict[MetricTag, list[type[TruLensMetric]]] = {
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
