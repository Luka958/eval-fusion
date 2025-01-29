from types import TracebackType

from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from phoenix.evals.evaluators import Record

from .llm import PhoenixProxyLLM
from .metrics import TAG_TO_METRIC_TYPES, PhoenixMetric


class PhoenixEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.llm = PhoenixProxyLLM(settings)

    def __enter__(self) -> 'PhoenixEvaluator':
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[PhoenixMetric]],
    ) -> list[EvaluationOutput]:
        evaluators = [metric_type(self.llm) for metric_type in metric_types]

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

    def evaluate_by_tag(
        self,
        inputs: list[EvaluationInput],
        tag: MetricTag,
    ) -> list[EvaluationOutput]:
        if tag is not None:
            metric_types = TAG_TO_METRIC_TYPES[tag]

        return self.evaluate(inputs, metric_types)

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        pass
