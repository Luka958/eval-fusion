from abc import ABC, abstractmethod

from eval_fusion_core.models import EvaluationInput, EvaluationOutput

from .embedding_model import EvalFusionBaseEmbeddingModel
from .llm import EvalFusionBaseLLM


class EvalFusionBaseEvaluator(ABC):
    embedding_model: EvalFusionBaseEmbeddingModel
    llm: EvalFusionBaseLLM

    @abstractmethod
    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        pass
