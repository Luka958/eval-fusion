from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature
from ragas.metrics import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    ResponseRelevancy,
)


class RagasMetric(EvalFusionBaseMetric):
    CONTEXT_ENTITY_RECALL = 'context_entity_recall'
    CONTEXT_PRECISION = 'context_precision'
    CONTEXT_RECALL = 'context_recall'
    FAITHFULNESS = 'faithfulness'
    NOISE_SENSITIVITY = 'noise_sensitivity'
    RESPONSE_RELEVANCY = 'response_relevancy'


RagasMetricType = type[
    ContextEntityRecall
    | ContextPrecision
    | ContextRecall
    | Faithfulness
    | NoiseSensitivity
    | ResponseRelevancy
]

METRIC_TO_TYPE: dict[RagasMetric, RagasMetricType] = {
    RagasMetric.CONTEXT_ENTITY_RECALL: ContextEntityRecall,
    RagasMetric.CONTEXT_PRECISION: ContextPrecision,
    RagasMetric.CONTEXT_RECALL: ContextRecall,
    RagasMetric.FAITHFULNESS: Faithfulness,
    RagasMetric.NOISE_SENSITIVITY: NoiseSensitivity,
    RagasMetric.RESPONSE_RELEVANCY: ResponseRelevancy,
}

FEATURE_TO_METRICS: dict[Feature, RagasMetric] = {
    Feature.INPUT: [
        RagasMetric.CONTEXT_PRECISION,
        RagasMetric.CONTEXT_RECALL,
        RagasMetric.NOISE_SENSITIVITY,
        RagasMetric.RESPONSE_RELEVANCY,
    ],
    Feature.OUTPUT: [
        RagasMetric.FAITHFULNESS,
        RagasMetric.NOISE_SENSITIVITY,
        RagasMetric.RESPONSE_RELEVANCY,
    ],
    Feature.GROUND_TRUTH: [
        RagasMetric.CONTEXT_ENTITY_RECALL,
        RagasMetric.CONTEXT_PRECISION,
        RagasMetric.CONTEXT_RECALL,
        RagasMetric.NOISE_SENSITIVITY,
    ],
    Feature.RELEVANT_CHUNKS: [
        RagasMetric.CONTEXT_ENTITY_RECALL,
        RagasMetric.CONTEXT_PRECISION,
        RagasMetric.CONTEXT_RECALL,
        RagasMetric.FAITHFULNESS,
        RagasMetric.NOISE_SENSITIVITY,
    ],
    Feature.ALL: [
        RagasMetric.CONTEXT_ENTITY_RECALL,
        RagasMetric.CONTEXT_PRECISION,
        RagasMetric.CONTEXT_RECALL,
        RagasMetric.FAITHFULNESS,
        RagasMetric.NOISE_SENSITIVITY,
        RagasMetric.RESPONSE_RELEVANCY,
    ],
}
