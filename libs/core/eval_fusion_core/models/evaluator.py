from pydantic import BaseModel

from eval_fusion_core.abstractions import EmbeddingModel, LargeLanguageModel
from eval_fusion_core.models import EvaluationInput, EvaluationOutput


class Evaluator(BaseModel):
    embedding_model: EmbeddingModel
    large_language_model: LargeLanguageModel

    def evaluate(inputs: list[EvaluationInput]) -> list[EvaluationOutput]:
        pass  # TODO how to accept metrics? add 'provider' attribute to Metric and group by provider
