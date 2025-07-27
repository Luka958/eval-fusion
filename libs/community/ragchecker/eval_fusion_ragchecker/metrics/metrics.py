from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature
from ragchecker.metrics import (
    claim_recall,
    context_precision,
    context_utilization,
    f1,
    faithfulness,
    hallucination,
    noise_sensitivity_in_irrelevant,
    noise_sensitivity_in_relevant,
    precision,
    recall,
    self_knowledge,
)


class RagCheckerMetric(EvalFusionBaseMetric):
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1'
    CLAIM_RECALL = 'claim_recall'
    CONTEXT_PRECISION = 'context_precision'
    CONTEXT_UTILIZATION = 'context_utilization'
    NOISE_SENSITIVITY_IN_RELEVANT = 'noise_sensitivity_in_relevant'
    NOISE_SENSITIVITY_IN_IRRELEVANT = 'noise_sensitivity_in_irrelevant'
    HALLUCINATION = 'hallucination'
    SELF_KNOWLEDGE = 'self_knowledge'
    FAITHFULNESS = 'faithfulness'


RagCheckerMetricType = str

METRIC_TO_TYPE: dict[RagCheckerMetric, RagCheckerMetricType] = {
    RagCheckerMetric.PRECISION: precision,
    RagCheckerMetric.RECALL: recall,
    RagCheckerMetric.F1: f1,
    RagCheckerMetric.CLAIM_RECALL: claim_recall,
    RagCheckerMetric.CONTEXT_PRECISION: context_precision,
    RagCheckerMetric.CONTEXT_UTILIZATION: context_utilization,
    RagCheckerMetric.NOISE_SENSITIVITY_IN_RELEVANT: noise_sensitivity_in_relevant,
    RagCheckerMetric.NOISE_SENSITIVITY_IN_IRRELEVANT: noise_sensitivity_in_irrelevant,
    RagCheckerMetric.HALLUCINATION: hallucination,
    RagCheckerMetric.SELF_KNOWLEDGE: self_knowledge,
    RagCheckerMetric.FAITHFULNESS: faithfulness,
}

FEATURE_TO_METRICS = {
    Feature.INPUT: [],
    Feature.OUTPUT: [
        RagCheckerMetric.CONTEXT_UTILIZATION,
        RagCheckerMetric.PRECISION,
        RagCheckerMetric.RECALL,
        RagCheckerMetric.F1,
        RagCheckerMetric.FAITHFULNESS,
        RagCheckerMetric.HALLUCINATION,
        RagCheckerMetric.NOISE_SENSITIVITY_IN_RELEVANT,
        RagCheckerMetric.NOISE_SENSITIVITY_IN_IRRELEVANT,
        RagCheckerMetric.SELF_KNOWLEDGE,
    ],
    Feature.GROUND_TRUTH: [
        RagCheckerMetric.CLAIM_RECALL,
        RagCheckerMetric.CONTEXT_PRECISION,
        RagCheckerMetric.CONTEXT_UTILIZATION,
        RagCheckerMetric.PRECISION,
        RagCheckerMetric.RECALL,
        RagCheckerMetric.F1,
        RagCheckerMetric.SELF_KNOWLEDGE,
    ],
    Feature.RELEVANT_CHUNKS: [
        RagCheckerMetric.CLAIM_RECALL,
        RagCheckerMetric.CONTEXT_PRECISION,
        RagCheckerMetric.CONTEXT_UTILIZATION,
        RagCheckerMetric.FAITHFULNESS,
        RagCheckerMetric.HALLUCINATION,
        RagCheckerMetric.NOISE_SENSITIVITY_IN_RELEVANT,
        RagCheckerMetric.NOISE_SENSITIVITY_IN_IRRELEVANT,
        RagCheckerMetric.SELF_KNOWLEDGE,
    ],
    Feature.ALL: [
        RagCheckerMetric.PRECISION,
        RagCheckerMetric.RECALL,
        RagCheckerMetric.F1,
        RagCheckerMetric.CLAIM_RECALL,
        RagCheckerMetric.CONTEXT_PRECISION,
        RagCheckerMetric.CONTEXT_UTILIZATION,
        RagCheckerMetric.NOISE_SENSITIVITY_IN_RELEVANT,
        RagCheckerMetric.NOISE_SENSITIVITY_IN_IRRELEVANT,
        RagCheckerMetric.HALLUCINATION,
        RagCheckerMetric.SELF_KNOWLEDGE,
        RagCheckerMetric.FAITHFULNESS,
    ],
}
