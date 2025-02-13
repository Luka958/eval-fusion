from __future__ import annotations

from eval_fusion_core.enums import Feature
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)


PhoenixMetric = HallucinationEvaluator | QAEvaluator | RelevanceEvaluator


TAG_TO_METRIC_TYPES: dict[Feature, list[type[PhoenixMetric]]] = {
    Feature.INPUT: [
        HallucinationEvaluator,
        QAEvaluator,
        RelevanceEvaluator,
    ],
    Feature.OUTPUT: [
        HallucinationEvaluator,
        QAEvaluator,
    ],
    Feature.GROUND_TRUTH: [],
    Feature.RELEVANT_CHUNKS: [
        HallucinationEvaluator,
        QAEvaluator,
        RelevanceEvaluator,
    ],
    Feature.ALL: [
        HallucinationEvaluator,
        QAEvaluator,
        RelevanceEvaluator,
    ],
}
