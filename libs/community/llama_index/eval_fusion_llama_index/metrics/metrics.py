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

LlamaIndexMetricUnion = (
    AnswerRelevancyEvaluator
    | ContextRelevancyEvaluator
    | CorrectnessEvaluator
    | RelevancyEvaluator
    | FaithfulnessEvaluator
    | SemanticSimilarityEvaluator
)

METRIC_TO_TYPE: dict[LlamaIndexMetric, LlamaIndexMetricType] = {
    LlamaIndexMetric.ANSWER_RELEVANCY: AnswerRelevancyEvaluator,
    LlamaIndexMetric.CONTEXT_RELEVANCY: ContextRelevancyEvaluator,
    LlamaIndexMetric.CORRECTNESS: CorrectnessEvaluator,
    LlamaIndexMetric.RELEVANCY: RelevancyEvaluator,
    LlamaIndexMetric.FAITHFULNESS: FaithfulnessEvaluator,
    LlamaIndexMetric.SEMANTIC_SIMILARITY: SemanticSimilarityEvaluator,
}

FEATURE_TO_METRICS = {
    Feature.INPUT: [
        LlamaIndexMetric.ANSWER_RELEVANCY,
        LlamaIndexMetric.CONTEXT_RELEVANCY,
        LlamaIndexMetric.CORRECTNESS,
        LlamaIndexMetric.RELEVANCY,
    ],
    Feature.OUTPUT: [
        LlamaIndexMetric.ANSWER_RELEVANCY,
        LlamaIndexMetric.CORRECTNESS,
        LlamaIndexMetric.RELEVANCY,
        LlamaIndexMetric.FAITHFULNESS,
        LlamaIndexMetric.SEMANTIC_SIMILARITY,
    ],
    Feature.GROUND_TRUTH: [
        LlamaIndexMetric.CORRECTNESS,
        LlamaIndexMetric.SEMANTIC_SIMILARITY,
    ],
    Feature.RELEVANT_CHUNKS: [
        LlamaIndexMetric.CONTEXT_RELEVANCY,
        LlamaIndexMetric.RELEVANCY,
        LlamaIndexMetric.FAITHFULNESS,
    ],
    Feature.ALL: [],
}
