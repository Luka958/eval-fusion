from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature
from tonic_validate.metrics import (
    AnswerConsistencyMetric,
    AnswerSimilarityMetric,
    AugmentationAccuracyMetric,
    AugmentationPrecisionMetric,
    RetrievalPrecisionMetric,
)


class TonicValidateMetric(EvalFusionBaseMetric):
    ANSWER_CONSISTENCY = 'answer_consistency'
    ANSWER_SIMILARITY = 'answer_similarity'
    AUGMENTATION_ACCURACY = 'augmentation_accuracy'
    AUGMENTATION_PRECISION = 'augmentation_precision'
    RETRIEVAL_PRECISION = 'retrieval_precision'


TonicValidateMetricUnion = (
    AnswerConsistencyMetric
    | AnswerSimilarityMetric
    | AugmentationAccuracyMetric
    | AugmentationPrecisionMetric
    | RetrievalPrecisionMetric
)
TonicValidateMetricType = type[TonicValidateMetricUnion]


METRIC_TO_TYPE: dict[TonicValidateMetric, TonicValidateMetricType] = {
    TonicValidateMetric.ANSWER_CONSISTENCY: AnswerConsistencyMetric,
    TonicValidateMetric.ANSWER_SIMILARITY: AnswerSimilarityMetric,
    TonicValidateMetric.AUGMENTATION_ACCURACY: AugmentationAccuracyMetric,
    TonicValidateMetric.AUGMENTATION_PRECISION: AugmentationPrecisionMetric,
    TonicValidateMetric.RETRIEVAL_PRECISION: RetrievalPrecisionMetric,
}

FEATURE_TO_METRICS = {
    Feature.INPUT: [
        TonicValidateMetric.AUGMENTATION_PRECISION,
        TonicValidateMetric.ANSWER_SIMILARITY,
        TonicValidateMetric.RETRIEVAL_PRECISION,
    ],
    Feature.OUTPUT: [
        TonicValidateMetric.ANSWER_CONSISTENCY,
        TonicValidateMetric.AUGMENTATION_ACCURACY,
        TonicValidateMetric.ANSWER_SIMILARITY,
        TonicValidateMetric.AUGMENTATION_PRECISION,
    ],
    Feature.GROUND_TRUTH: [TonicValidateMetric.ANSWER_SIMILARITY],
    Feature.RELEVANT_CHUNKS: [
        TonicValidateMetric.ANSWER_CONSISTENCY,
        TonicValidateMetric.AUGMENTATION_ACCURACY,
        TonicValidateMetric.AUGMENTATION_PRECISION,
        TonicValidateMetric.RETRIEVAL_PRECISION,
    ],
    Feature.ALL: [
        TonicValidateMetric.ANSWER_CONSISTENCY,
        TonicValidateMetric.ANSWER_SIMILARITY,
        TonicValidateMetric.AUGMENTATION_ACCURACY,
        TonicValidateMetric.AUGMENTATION_PRECISION,
        TonicValidateMetric.RETRIEVAL_PRECISION,
    ],
}
