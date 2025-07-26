from __future__ import annotations

import asyncio
import os

from time import perf_counter
from types import TracebackType
from typing import NamedTuple

from deepeval.test_case import LLMTestCase
from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import Feature
from eval_fusion_core.exceptions import EvalFusionException
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings

from .llm import DeepEvalProxyLLM
from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    DeepEvalMetric,
    DeepEvalMetricType,
    DeepEvalMetricUnion,
)


class DeepEvalEvaluationTask(NamedTuple):
    test_case_id: int
    test_case: LLMTestCase
    metric_id: int
    metric: DeepEvalMetricUnion


class DeepEvalEvaluationTaskResult(NamedTuple):
    score: float | None
    reason: str | None
    error: str | None
    time: float


class DeepEvalEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = DeepEvalProxyLLM(settings)

    def __enter__(self) -> DeepEvalEvaluator:
        os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = 'YES'

        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[DeepEvalMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [
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

            for metric in metric_instances:
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
        metrics: list[DeepEvalMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [
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

        metric_type_to_tasks: dict[
            DeepEvalMetricType, list[DeepEvalEvaluationTask]
        ] = {}

        for i, test_case in enumerate(test_cases):
            for j, metric in enumerate(metric_instances):
                task = DeepEvalEvaluationTask(
                    test_case_id=i,
                    test_case=test_case,
                    metric_id=j,
                    metric=metric,
                )
                metric_type_to_tasks.setdefault(type(metric), []).append(task)

        ids_to_entry: dict[tuple[int, int], EvaluationOutputEntry] = {}

        for _, tasks in metric_type_to_tasks.items():
            coros = [self._run_task(task) for task in tasks]
            batch = await asyncio.gather(*coros)

            for task, result in zip(tasks, batch):
                i, _, j, metric = task
                score, reason, error, time = result
                ids_to_entry[(i, j)] = EvaluationOutputEntry(
                    metric_name=str(metric.__name__),
                    score=score,
                    reason=reason,
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
        task: DeepEvalEvaluationTask,
    ) -> DeepEvalEvaluationTaskResult:
        try:
            start = perf_counter()
            score = await task.metric.a_measure(task.test_case, _show_indicator=False)
            time = perf_counter() - start

            return DeepEvalEvaluationTaskResult(
                score=score,
                reason=task.metric.reason,
                error=None,
                time=time,
            )

        except Exception as e:
            time = perf_counter() - start

            return DeepEvalEvaluationTaskResult(
                score=None,
                reason=None,
                error=str(e),
                time=time,
            )

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = self._llm.get_token_usage()
