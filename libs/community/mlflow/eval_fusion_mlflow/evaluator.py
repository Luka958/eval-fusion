from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)

from .llm import MlFlowProxyLLM


class MlFlowEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.llm: MlFlowProxyLLM = MlFlowProxyLLM(llm_delegate=llm)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics
        pass
