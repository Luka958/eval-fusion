from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)


class PhoenixMetric(EvalFusionBaseMetric):
    HALLUCINATION = 'hallucination'
    QA = 'qa'
    RELEVANCE = 'relevance'


PhoenixMetricType = type[HallucinationEvaluator | QAEvaluator | RelevanceEvaluator]

METRIC_TO_TYPE: dict[PhoenixMetric, PhoenixMetricType] = {
    PhoenixMetric.HALLUCINATION: HallucinationEvaluator,
    PhoenixMetric.QA: QAEvaluator,
    PhoenixMetric.RELEVANCE: RelevanceEvaluator,
}

FEATURE_TO_METRICS = {
    Feature.INPUT: [
        PhoenixMetric.HALLUCINATION,
        PhoenixMetric.QA,
        PhoenixMetric.RELEVANCE,
    ],
    Feature.OUTPUT: [
        PhoenixMetric.HALLUCINATION,
        PhoenixMetric.QA,
    ],
    Feature.GROUND_TRUTH: [],
    Feature.RELEVANT_CHUNKS: [
        PhoenixMetric.HALLUCINATION,
        PhoenixMetric.QA,
        PhoenixMetric.RELEVANCE,
    ],
    Feature.ALL: [
        PhoenixMetric.HALLUCINATION,
        PhoenixMetric.QA,
        PhoenixMetric.RELEVANCE,
    ],
}
