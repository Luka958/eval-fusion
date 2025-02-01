from __future__ import annotations

from eval_fusion_core.enums import MetricTag
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


TAG_TO_METRIC_TYPES: dict[MetricTag, list[type[RagasMetric]]] = {
    MetricTag.INPUT: [
        ContextPrecision,
        ContextRecall,
        NoiseSensitivity,
        ResponseRelevancy,
    ],
    MetricTag.OUTPUT: [
        Faithfulness,
        NoiseSensitivity,
        ResponseRelevancy,
    ],
    MetricTag.GROUND_TRUTH: [
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        NoiseSensitivity,
    ],
    MetricTag.RELEVANT_CHUNKS: [
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        NoiseSensitivity,
    ],
    MetricTag.ALL: [
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        NoiseSensitivity,
        ResponseRelevancy,
    ],
}
