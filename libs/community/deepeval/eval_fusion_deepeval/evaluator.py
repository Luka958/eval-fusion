import os

from time import perf_counter
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
        self._llm = DeepEvalProxyLLM(settings)

    def __enter__(self) -> 'DeepEvalEvaluator':
        os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = 'YES'

        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[DeepEvalMetric]],
    ) -> list[EvaluationOutput]:
        metrics = [
            metric_type(
                threshold=0.5,
                model=self._llm,
                include_reason=True,
                async_mode=False,
                strict_mode=False,
                verbose_mode=False,
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

        outputs: list[EvaluationOutput] = []

        for i, test_case in enumerate(test_cases):
            output_entries: list[EvaluationOutputEntry] = []

            for metric in metrics:
                metric_name = str(metric.__name__)

                try:
                    start = perf_counter()
                    score = metric.measure(test_case, _show_indicator=False)
                    time = perf_counter() - start

                    reason = metric.reason

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
        self.token_usage = self._llm.get_token_usage()
