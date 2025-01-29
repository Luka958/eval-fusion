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
from ragas.metrics import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    ResponseRelevancy,
    SingleTurnMetric,
)

from .em import RagasProxyEM
from .llm import RagasProxyLLM
from .metrics import TAG_TO_METRICS


class RagasEvaluator(EvalFusionBaseEvaluator):
    def __init__(
        self, llm_settings: EvalFusionLLMSettings, em_settings: EvalFusionEMSettings
    ):
        self.llm = RagasProxyLLM(llm_settings)
        self.em = RagasProxyEM(em_settings)

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[SingleTurnMetric]],
        tag: MetricTag | None = None,
    ) -> list[EvaluationOutput]:
        if tag is not None:
            metric_types = TAG_TO_METRICS[tag]

        context_precision = ContextPrecision(llm=self.llm)
        context_recall = ContextRecall(llm=self.llm)
        context_entity_recall = ContextEntityRecall(llm=self.llm)
        noise_sensitivity = NoiseSensitivity(llm=self.llm)
        response_relevancy = ResponseRelevancy(llm=self.llm, embeddings=self.em)
        faithfulness = Faithfulness(llm=self.llm)

        metrics: list[SingleTurnMetric] = [
            metric_type() for metric_type in metric_types
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
