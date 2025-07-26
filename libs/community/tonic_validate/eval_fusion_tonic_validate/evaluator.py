from __future__ import annotations

import asyncio

from time import perf_counter
from types import TracebackType
from typing import NamedTuple

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

from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    TonicValidateMetric,
    TonicValidateMetricType,
)


class TonicValidateEvaluationTask(NamedTuple):
    llm_response_id: int
    llm_response: LLMResponse
    scorer_id: int
    scorer: ValidateScorer


class TonicValidateEvaluationTaskResult(NamedTuple):
    score: float | None
    error: str | None
    time: float


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

        metric_type_to_tasks: dict[
            TonicValidateMetricType, list[TonicValidateEvaluationTask]
        ] = {}

        for i, llm_response in enumerate(llm_responses):
            for j, scorer in enumerate(scorers):
                task = TonicValidateEvaluationTask(
                    llm_response_id=i,
                    llm_response=llm_response,
                    scorer_id=j,
                    scorer=scorer,
                )
                metric_type_to_tasks.setdefault(type(scorer), []).append(task)

        ids_to_entry: dict[tuple[int, int], EvaluationOutputEntry] = {}

        for _, tasks in metric_type_to_tasks.items():
            coros = [self._run_task(task) for task in tasks]
            batch = await asyncio.gather(*coros)

            for task, result in zip(tasks, batch):
                i, _, j, scorer = task
                score, error, time = result
                ids_to_entry[(i, j)] = EvaluationOutputEntry(
                    metric_name=scorer.metrics[0].name,
                    score=score,
                    reason=None,
                    error=error,
                    time=time,
                )

        outputs: list[EvaluationOutput] = []

        for i, x in enumerate(inputs):
            entries = [ids_to_entry[(i, j)] for j in range(len(metric_instances))]
            outputs.append(EvaluationOutput(input_id=x.id, output_entries=entries))

        return outputs

    async def _run_task(
        self,
        task: TonicValidateEvaluationTask,
    ) -> TonicValidateEvaluationTaskResult:
        try:
            start = perf_counter()
            metric_name = task.scorer.metrics[0].name
            run = await task.scorer.a_score_responses(
                responses=[task.llm_response], parallelism=1
            )
            score = run.overall_scores[metric_name]
            time = perf_counter() - start

            return TonicValidateEvaluationTaskResult(score=score, error=None, time=time)

        except Exception as e:
            time = perf_counter() - start

            return TonicValidateEvaluationTaskResult(
                score=None, error=str(e), time=time
            )

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
        self.token_usage = None
