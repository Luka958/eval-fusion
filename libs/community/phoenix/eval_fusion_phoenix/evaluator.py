from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from phoenix.evals import (
    HallucinationEvaluator,
    LLMEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)
from phoenix.evals.evaluators import Record

from .llm import PhoenixProxyLLM


class PhoenixEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.llm: PhoenixProxyLLM = PhoenixProxyLLM(llm=llm)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        evaluators: list[LLMEvaluator] = [
            HallucinationEvaluator(self.llm),
            QAEvaluator(self.llm),
            RelevanceEvaluator(self.llm),
        ]

        records: list[Record] = [
            {
                'input': x.input,
                'output': x.output,
                'reference': '\n\n'.join(x.relevant_chunks),
            }
            for x in inputs
        ]

        return [
            EvaluationOutput(
                input_id=inputs[i].id,
                output_entries=[
                    EvaluationOutputEntry(
                        metric_name=evaluator.__class__.__name__.lower().removesuffix(
                            'evaluator'
                        ),
                        score=score,
                        reason=reason,
                    )
                    for evaluator in evaluators
                    for _, score, reason in [
                        evaluator.evaluate(record, provide_explanation=True)
                    ]
                ],
            )
            for i, record in enumerate(records)
        ]
