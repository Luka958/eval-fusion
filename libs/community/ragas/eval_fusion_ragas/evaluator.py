from eval_fusion_core.base import (
    EvalFusionBaseEM,
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
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


class RagasEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM, em: EvalFusionBaseEM):
        self.llm: RagasProxyLLM = RagasProxyLLM(llm=llm)
        self.em = RagasProxyEM(em=em)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        context_precision = ContextPrecision(llm=self.llm)
        context_recall = ContextRecall(llm=self.llm)
        context_entity_recall = ContextEntityRecall(llm=self.llm)
        noise_sensitivity = NoiseSensitivity(llm=self.llm)
        response_relevancy = ResponseRelevancy(llm=self.llm, embeddings=self.em)
        faithfulness = Faithfulness(llm=self.llm)

        metrics: list[SingleTurnMetric] = [
            context_precision,
            context_recall,
            context_entity_recall,
            noise_sensitivity,
            response_relevancy,
            faithfulness,
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
