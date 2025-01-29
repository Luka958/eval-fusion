from types import TracebackType

from deepeval.test_case import LLMTestCase
from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings

from .llm import DeepEvalProxyLLM
from .metrics import TAG_TO_METRIC_TYPES, DeepEvalMetric


class DeepEvalEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.llm = DeepEvalProxyLLM(settings)

    def __enter__(self) -> 'DeepEvalEvaluator':
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[DeepEvalMetric]],
    ) -> list[EvaluationOutput]:
        metrics = [
            metric_type(
                threshold=0.5,
                model=self.llm,
                include_reason=True,
                async_mode=False,
                strict_mode=False,
                verbose_mode=False,
                _show_indicator=False,
            )
            for metric_type in metric_types
        ]

        test_cases = [
            LLMTestCase(
                input=x.input,
                actual_output=x.output,
                expected_output=x.ground_truth,
                context=None,
                retrieval_context=x.relevant_chunks,
                tools_called=None,
                expected_tools=None,
            )
            for x in inputs
        ]

        evaluation_outputs: list[EvaluationOutput] = []

        for i, test_case in enumerate(test_cases):
            evaluation_output_entires: list[EvaluationOutputEntry] = []

            for metric in metrics:
                metric_name = metric.__name__

                try:
                    score = metric.measure(test_case)
                    reason = metric.reason

                    evaluation_output_entires.append(
                        EvaluationOutputEntry(
                            metric_name=metric_name,
                            score=score,
                            reason=reason,
                        )
                    )

                except Exception as e:
                    evaluation_output_entires.append(
                        EvaluationOutputEntry(metric_name=metric_name, error=e)
                    )

            evaluation_outputs.append(
                EvaluationOutput(
                    input_id=inputs[i].id,
                    output_entries=evaluation_output_entires,
                )
            )

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
