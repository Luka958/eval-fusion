from __future__ import annotations

from eval_fusion_core.enums import Feature
from ragas.metrics import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    ResponseRelevancy,
)


RagasMetric = (
    ContextEntityRecall
    | ContextPrecision
    | ContextRecall
    | Faithfulness
    | NoiseSensitivity
    | ResponseRelevancy
)


TAG_TO_METRIC_TYPES: dict[Feature, list[type[RagasMetric]]] = {
    Feature.INPUT: [
        ContextPrecision,
        ContextRecall,
        NoiseSensitivity,
        ResponseRelevancy,
    ],
    Feature.OUTPUT: [
        Faithfulness,
        NoiseSensitivity,
        ResponseRelevancy,
    ],
    Feature.GROUND_TRUTH: [
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        NoiseSensitivity,
    ],
    Feature.RELEVANT_CHUNKS: [
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        NoiseSensitivity,
    ],
    Feature.ALL: [
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        NoiseSensitivity,
        ResponseRelevancy,
    ],
}
