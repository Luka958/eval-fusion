from __future__ import annotations

from time import perf_counter
from types import TracebackType

from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import Feature
from eval_fusion_core.exceptions import EvalFusionException
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from tonic_validate import BenchmarkItem, LLMResponse, ValidateScorer

from .metrics import FEATURE_TO_METRICS, METRIC_TO_TYPE, TonicValidateMetric


class TonicValidateEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm_settings: EvalFusionLLMSettings):
        self._llm = llm_settings.kwargs.get('model_name')

        if self._llm is None:
            raise ValueError()

    def __enter__(self) -> TonicValidateEvaluator:
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[TonicValidateMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [metric_type() for metric_type in metric_types]
        scorers = [
            ValidateScorer([metric_instance], model_evaluator=self._llm)
            for metric_instance in metric_instances
        ]

        llm_responses = [
            LLMResponse(
                llm_answer=x.output,
                llm_context_list=x.relevant_chunks,
                benchmark_item=BenchmarkItem(question=x.input, answer=x.ground_truth),
            )
            for x in inputs
        ]

        outputs: list[EvaluationOutput] = []

        for i, llm_response in enumerate(llm_responses):
            output_entries: list[EvaluationOutputEntry] = []

            for scorer in scorers:
                metric_name = scorer.metrics[0].name

                try:
                    start = perf_counter()
                    run = scorer.score_responses(
                        responses=[llm_response], parallelism=1
                    )
                    score = run.overall_scores[metric_name]
                    time = perf_counter() - start

                    output_entries.append(
                        EvaluationOutputEntry(
                            metric_name=metric_name,
                            score=score,
                            reason=None,
                            error=None,
                            time=time,
                        )
                    )

                except Exception as e:
                    time = perf_counter() - start

                    output_entries.append(
                        EvaluationOutputEntry(
                            metric_name=metric_name,
                            score=None,
                            reason=None,
                            error=str(e),
                            time=time,
                        )
                    )

            outputs.append(
                EvaluationOutput(
                    input_id=inputs[i].id,
                    output_entries=output_entries,
                )
            )

        return outputs

    async def a_evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[TonicValidateMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        pass

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
        self.token_usage = None
