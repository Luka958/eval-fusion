from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature


class TruLensMetric(EvalFusionBaseMetric):
    CONTEXT_RELEVANCE = 'context_relevance'
    GROUNDEDNESS = 'groundedness'
    RELEVANCE = 'relevance'


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
