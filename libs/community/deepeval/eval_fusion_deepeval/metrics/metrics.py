from __future__ import annotations

from enum import Enum

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from eval_fusion_core.enums import MetricTag


class DeepEvalMetric(str, Enum):
    ANSWER_RELEVANCY = 'answer_relevancy'
    CONTEXTUAL_PRECISION = 'contextual_precision'
    CONTEXTUAL_RECALL = 'contextual_recall'
    CONTEXTUAL_RELEVANCY = 'contextual_relevancy'
    FAITHFULNESS = 'faithfulness'


DeepEvalMetricType = type[
    AnswerRelevancyMetric
    | ContextualPrecisionMetric
    | ContextualRecallMetric
    | ContextualRelevancyMetric
    | FaithfulnessMetric
]

METRIC_TO_TYPE: dict[DeepEvalMetric, DeepEvalMetricType] = {
    DeepEvalMetric.ANSWER_RELEVANCY: AnswerRelevancyMetric,
    DeepEvalMetric.CONTEXTUAL_PRECISION: ContextualPrecisionMetric,
    DeepEvalMetric.CONTEXTUAL_RECALL: ContextualRelevancyMetric,
    DeepEvalMetric.CONTEXTUAL_RELEVANCY: ContextualRelevancyMetric,
    DeepEvalMetric.FAITHFULNESS: FaithfulnessMetric,
}

TAG_TO_METRICS = {
    MetricTag.INPUT: [
        DeepEvalMetric.ANSWER_RELEVANCY,
        DeepEvalMetric.CONTEXTUAL_PRECISION,
        DeepEvalMetric.CONTEXTUAL_RECALL,
        DeepEvalMetric.CONTEXTUAL_RELEVANCY,
        DeepEvalMetric.FAITHFULNESS,
    ],
    MetricTag.OUTPUT: [
        DeepEvalMetric.ANSWER_RELEVANCY,
        DeepEvalMetric.CONTEXTUAL_PRECISION,
        DeepEvalMetric.CONTEXTUAL_RECALL,
        DeepEvalMetric.CONTEXTUAL_RELEVANCY,
        DeepEvalMetric.FAITHFULNESS,
    ],
    MetricTag.GROUND_TRUTH: [
        DeepEvalMetric.CONTEXTUAL_PRECISION,
        DeepEvalMetric.CONTEXTUAL_RECALL,
    ],
    MetricTag.RELEVANT_CHUNKS: [
        DeepEvalMetric.CONTEXTUAL_PRECISION,
        DeepEvalMetric.CONTEXTUAL_RECALL,
        DeepEvalMetric.CONTEXTUAL_RELEVANCY,
        DeepEvalMetric.FAITHFULNESS,
    ],
    MetricTag.ALL: [
        DeepEvalMetric.ANSWER_RELEVANCY,
        DeepEvalMetric.CONTEXTUAL_PRECISION,
        DeepEvalMetric.CONTEXTUAL_RECALL,
        DeepEvalMetric.CONTEXTUAL_RELEVANCY,
        DeepEvalMetric.FAITHFULNESS,
    ],
}
