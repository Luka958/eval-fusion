from eval_fusion_core.base import (
    EvalFusionBaseEM,
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionEMSettings, EvalFusionLLMSettings
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy

from .em import RagasProxyEM
from .llm import RagasProxyLLM
from .metrics import TAG_TO_METRICS, RagasMetric


class RagasEvaluator(EvalFusionBaseEvaluator):
    def __init__(
        self, llm_settings: EvalFusionLLMSettings, em_settings: EvalFusionEMSettings
    ):
        self.llm = RagasProxyLLM(llm_settings)
        self.em = RagasProxyEM(em_settings)

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[RagasMetric]],
        tag: MetricTag | None = None,
    ) -> list[EvaluationOutput]:
        if tag is not None:
            metric_types = TAG_TO_METRICS[tag]

        metrics = [
            metric_type(llm=self.llm, embeddings=self.em)
            if metric_type == ResponseRelevancy
            else metric_type(llm=self.llm)
            for metric_type in metric_types
        ]

        single_turn_samples = [
            SingleTurnSample(
                user_input=x.input,
                retrieved_contexts=x.relevant_chunks,
                reference_contexts=None,
                response=x.output,
                multi_responses=None,
                reference=x.ground_truth,
                rubrics=None,
            )
            for x in inputs
        ]

        return [
            EvaluationOutput(
                input_id=inputs[i].id,
                output_entries=[
                    EvaluationOutputEntry(
                        metric_name=metric.name,
                        score=metric.single_turn_score(single_turn_sample),
                        reason=None,
                    )
                    for metric in metrics
                ],
            )
            for i, single_turn_sample in enumerate(single_turn_samples)
        ]
