from types import MappingProxyType

from eval_fusion_core.enums import MetricTag
from ragas.metrics import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    ResponseRelevancy,
    SingleTurnMetric,
)


TAGS_TO_METRICS: dict[MetricTag, list[type[SingleTurnMetric]]] = MappingProxyType(
    {
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
)
