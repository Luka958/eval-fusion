from types import MappingProxyType
from typing import Union

from eval_fusion_core.enums import MetricTag
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)


PhoenixMetric = Union[
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
]


TAG_TO_METRIC_TYPES: dict[MetricTag, list[type[PhoenixMetric]]] = MappingProxyType(
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
