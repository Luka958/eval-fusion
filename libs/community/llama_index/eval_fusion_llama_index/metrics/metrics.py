from __future__ import annotations

from eval_fusion_core.base import EvalFusionBaseMetric
from eval_fusion_core.enums import Feature
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)


class LlamaIndexMetric(EvalFusionBaseMetric):
    ANSWER_RELEVANCY = 'answer_relevancy'
    CONTEXT_RELEVANCY = 'context_relevancy'
    CORRECTNESS = 'correctness'
    RELEVANCY = 'relevancy'
    FAITHFULNESS = 'faithfulness'
    SEMANTIC_SIMILARITY = 'semantic_similarity'


LlamaIndexMetricType = type[
    AnswerRelevancyEvaluator
    | ContextRelevancyEvaluator
    | CorrectnessEvaluator
    | RelevancyEvaluator
    | FaithfulnessEvaluator
    | SemanticSimilarityEvaluator
]

METRIC_TO_TYPE: dict[LlamaIndexMetric, LlamaIndexMetricType] = {
    LlamaIndexMetric.ANSWER_RELEVANCY: AnswerRelevancyEvaluator,
    LlamaIndexMetric.CONTEXT_RELEVANCY: ContextRelevancyEvaluator,
    LlamaIndexMetric.CORRECTNESS: CorrectnessEvaluator,
    LlamaIndexMetric.RELEVANCY: RelevancyEvaluator,
    LlamaIndexMetric.FAITHFULNESS: FaithfulnessEvaluator,
    LlamaIndexMetric.SEMANTIC_SIMILARITY: SemanticSimilarityEvaluator,
}

FEATURE_TO_METRICS = {
    Feature.INPUT: [...],
    Feature.OUTPUT: [...],
    Feature.GROUND_TRUTH: [...],
    Feature.RELEVANT_CHUNKS: [...],
    Feature.ALL: [...],
}
