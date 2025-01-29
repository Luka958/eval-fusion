from enum import Enum
from types import MappingProxyType

from eval_fusion_core.enums import MetricTag


class LLMProviderType(str, Enum):
    CONTEXT_RELEVANCE = 'context_relevance'
    GROUNDEDNESS = 'groundedness'
    ANSWER_RELEVANCE = 'answer_relevance'


TAG_TO_METRIC_TYPES: dict[MetricTag, list[type[LLMProviderType]]] = MappingProxyType(
    {
        MetricTag.INPUT: [
            LLMProviderType.CONTEXT_RELEVANCE,
            LLMProviderType.ANSWER_RELEVANCE,
        ],
        MetricTag.OUTPUT: [
            LLMProviderType.CONTEXT_RELEVANCE,
            LLMProviderType.GROUNDEDNESS,
            LLMProviderType.ANSWER_RELEVANCE,
        ],
        MetricTag.GROUND_TRUTH: [],
        MetricTag.RELEVANT_CHUNKS: [
            LLMProviderType.CONTEXT_RELEVANCE,
            LLMProviderType.GROUNDEDNESS,
        ],
        MetricTag.ALL: [
            LLMProviderType.CONTEXT_RELEVANCE,
            LLMProviderType.GROUNDEDNESS,
            LLMProviderType.ANSWER_RELEVANCE,
        ],
    }
)
