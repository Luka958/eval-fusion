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
from phoenix.evals.evaluators import Record

from .llm import PhoenixProxyLLM
from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    PhoenixMetric,
    PhoenixMetricType,
    PhoenixMetricUnion,
)


class PhoenixEvaluationTask(NamedTuple):
    record_id: int
    record: Record
    evaluator_id: int
    evaluator: PhoenixMetricUnion


class PhoenixEvaluationTaskResult(NamedTuple):
    score: float | None
    reason: str | None
    error: str | None
    time: float


class PhoenixEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = PhoenixProxyLLM(settings=settings)

    def __enter__(self) -> PhoenixEvaluator:
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[PhoenixMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
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

    async def a_evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[PhoenixMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        evaluators = [metric_type(self._llm) for metric_type in metric_types]

        records: list[Record] = [
            {
                'input': x.input,
                'output': x.output,
                'reference': '\n\n'.join(x.relevant_chunks),
            }
            for x in inputs
        ]

        # --------
        metric_type_to_tasks: dict[PhoenixMetricType, list[PhoenixEvaluationTask]] = {}

        for i, record in enumerate(records):
            for j, evaluator in enumerate(evaluators):
                task = PhoenixEvaluationTask(
                    record_id=i,
                    record=record,
                    evaluator_id=j,
                    evaluator=evaluator,
                )
                metric_type_to_tasks.setdefault(type(evaluator), []).append(task)

        ids_to_entry: dict[tuple[int, int], EvaluationOutputEntry] = {}

        for _, tasks in metric_type_to_tasks.items():
            coros = [self._run_task(task) for task in tasks]
            batch = await asyncio.gather(*coros)

            for task, result in zip(tasks, batch):
                i, _, j, evaluator = task
                score, reason, error, time = result
                metric_name = evaluator.__class__.__name__.lower().removesuffix(
                    'evaluator'
                )
                ids_to_entry[(i, j)] = EvaluationOutputEntry(
                    metric_name=metric_name,
                    score=score,
                    reason=reason,
                    error=error,
                    time=time,
                )

        outputs: list[EvaluationOutput] = []

        for i, x in enumerate(inputs):
            entries = [ids_to_entry[(i, j)] for j in range(len(evaluators))]
            outputs.append(EvaluationOutput(input_id=x.id, output_entries=entries))

        return outputs

    async def _run_task(
        self,
        task: PhoenixEvaluationTask,
    ) -> PhoenixEvaluationTaskResult:
        try:
            start = perf_counter()
            _, score, reason = await task.evaluator.aevaluate(
                task.record, provide_explanation=True
            )
            time = perf_counter() - start

            return PhoenixEvaluationTaskResult(
                score=score,
                reason=reason,
                error=None,
                time=time,
            )

        except Exception as e:
            time = perf_counter() - start

            return PhoenixEvaluationTaskResult(
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
