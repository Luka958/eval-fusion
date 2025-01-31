from time import perf_counter
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
        self._llm = PhoenixProxyLLM(settings=settings)

    def __enter__(self) -> 'PhoenixEvaluator':
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[PhoenixMetric]],
    ) -> list[EvaluationOutput]:
        evaluators = [metric_type(self._llm) for metric_type in metric_types]

        records: list[Record] = [
            {
                'input': x.input,
                'output': x.output,
                'reference': '\n\n'.join(x.relevant_chunks),
            }
            for x in inputs
        ]

        outputs: list[EvaluationOutput] = []

        for i, record in enumerate(records):
            output_entries: list[EvaluationOutputEntry] = []

            for evaluator in evaluators:
                metric_name = evaluator.__class__.__name__.lower().removesuffix(
                    'evaluator'
                )

                try:
                    start = perf_counter()
                    _, score, reason = evaluator.evaluate(
                        record, provide_explanation=True
                    )
                    time = perf_counter() - start

                    output_entries.append(
                        EvaluationOutputEntry(
                            metric_name=metric_name,
                            score=score,
                            reason=reason,
                            error=None,
                            time=time,
                        )
                    )

                except Exception as e:
                    output_entries.append(
                        EvaluationOutputEntry(
                            metric_name=metric_name,
                            score=None,
                            reason=None,
                            error=str(e),
                            time=None,
                        )
                    )

            outputs.append(
                EvaluationOutput(
                    input_id=inputs[i].id,
                    output_entries=output_entries,
                )
            )

        return outputs

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
