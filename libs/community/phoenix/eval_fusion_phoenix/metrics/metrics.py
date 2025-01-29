from types import MappingProxyType

from eval_fusion_core.enums import MetricTag
from phoenix.evals import (
    HallucinationEvaluator,
    LLMEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)


TAGS_TO_METRICS: dict[MetricTag, list[type[LLMEvaluator]]] = MappingProxyType(
    {
        MetricTag.INPUT: [
            HallucinationEvaluator,
            QAEvaluator,
            RelevanceEvaluator,
        ],
        MetricTag.OUTPUT: [
            HallucinationEvaluator,
            QAEvaluator,
        ],
        MetricTag.GROUND_TRUTH: [],
        MetricTag.RELEVANT_CHUNKS: [
            HallucinationEvaluator,
            QAEvaluator,
            RelevanceEvaluator,
        ],
        MetricTag.ALL: [
            HallucinationEvaluator,
            QAEvaluator,
            RelevanceEvaluator,
        ],
    }
)
