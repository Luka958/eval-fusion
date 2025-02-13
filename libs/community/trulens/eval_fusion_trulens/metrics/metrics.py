from __future__ import annotations

from eval_fusion_core.enums import Feature


class ContextRelevance:
    pass


class Groundedness:
    pass


class Relevance:
    pass


TruLensMetric = ContextRelevance | Groundedness | Relevance


TAG_TO_METRIC_TYPES: dict[Feature, list[type[TruLensMetric]]] = {
    Feature.INPUT: [
        ContextRelevance,
        Relevance,
    ],
    Feature.OUTPUT: [
        ContextRelevance,
        Groundedness,
        Relevance,
    ],
    Feature.GROUND_TRUTH: [],
    Feature.RELEVANT_CHUNKS: [
        ContextRelevance,
        Groundedness,
    ],
    Feature.ALL: [
        ContextRelevance,
        Groundedness,
        Relevance,
    ],
}
