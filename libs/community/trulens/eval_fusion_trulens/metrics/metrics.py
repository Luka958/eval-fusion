from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature


class ContextRelevance:
    pass


class Groundedness:
    pass


class Relevance:
    pass


class TruLensMetric(EvalFusionBaseMetric):
    CONTEXT_RELEVANCE = 'context_relevance'
    GROUNDEDNESS = 'groundedness'
    RELEVANCE = 'relevance'


TruLensMetricType = type[ContextRelevance | Groundedness | Relevance]


METRIC_TO_TYPE: dict[TruLensMetric, TruLensMetricType] = {
    TruLensMetric.CONTEXT_RELEVANCE: ContextRelevance,
    TruLensMetric.GROUNDEDNESS: Groundedness,
    TruLensMetric.RELEVANCE: Relevance,
}


FEATURE_TO_METRICS = {
    Feature.INPUT: [
        TruLensMetric.CONTEXT_RELEVANCE,
        TruLensMetric.RELEVANCE,
    ],
    Feature.OUTPUT: [
        TruLensMetric.CONTEXT_RELEVANCE,
        TruLensMetric.GROUNDEDNESS,
        TruLensMetric.RELEVANCE,
    ],
    Feature.GROUND_TRUTH: [],
    Feature.RELEVANT_CHUNKS: [
        TruLensMetric.CONTEXT_RELEVANCE,
        TruLensMetric.GROUNDEDNESS,
    ],
    Feature.ALL: [
        TruLensMetric.CONTEXT_RELEVANCE,
        TruLensMetric.GROUNDEDNESS,
        TruLensMetric.RELEVANCE,
    ],
}
