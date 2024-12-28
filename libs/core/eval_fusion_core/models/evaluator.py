from pydantic import BaseModel

from eval_fusion_core.abstractions import LLM, EmbeddingModel

from .input import EvaluationInput
from .output import EvaluationOutput


class Evaluator(BaseModel):
    # embedding_model: EmbeddingModel
    # large_language_model: LargeLanguageModel

    def evaluate(self, inputs: list, metrics: list) -> list:
        # TODO how to accept metrics? add 'provider' attribute to Metric and group by provider
        # TODO how to provide framework, NOTE: core doesn't know about frameworks
        pass
