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
from eval_fusion_core.models.settings import EvalFusionEMSettings, EvalFusionLLMSettings
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy

from .em import RagasProxyEM
from .llm import RagasProxyLLM
from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    RagasMetric,
    RagasMetricType,
    RagasMetricUnion,
)


class RagasEvaluationTask(NamedTuple):
    sample_id: int
    sample: SingleTurnSample
    metric_id: int
    metric: RagasMetricUnion


class RagasEvaluationTaskResult(NamedTuple):
    score: float | None
    error: str | None
    time: float


class RagasEvaluator(EvalFusionBaseEvaluator):
    def __init__(
        self, llm_settings: EvalFusionLLMSettings, em_settings: EvalFusionEMSettings
    ):
        self._llm = RagasProxyLLM(llm_settings)
        self._em = RagasProxyEM(em_settings)

    def __enter__(self) -> RagasEvaluator:
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[RagasMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [
            metric_type(llm=self._llm, embeddings=self._em)
            if metric_type == ResponseRelevancy
            else metric_type(llm=self._llm)
            for metric_type in metric_types
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

        outputs: list[EvaluationOutput] = []

        for i, single_turn_sample in enumerate(single_turn_samples):
            output_entries: list[EvaluationOutputEntry] = []

            for metric in metric_instances:
                metric_name = metric.name

                try:
                    start = perf_counter()
                    score = metric.single_turn_score(single_turn_sample)
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
        metrics: list[RagasMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [
            metric_type(llm=self._llm, embeddings=self._em)
            if metric_type == ResponseRelevancy
            else metric_type(llm=self._llm)
            for metric_type in metric_types
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

        metric_type_to_tasks: dict[RagasMetricType, list[RagasEvaluationTask]] = {}

        for i, sample in enumerate(single_turn_samples):
            for j, metric in enumerate(metric_instances):
                task = RagasEvaluationTask(
                    sample_id=i,
                    sample=sample,
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
                score, error, time = result
                ids_to_entry[(i, j)] = EvaluationOutputEntry(
                    metric_name=metric.name,
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
        task: RagasEvaluationTask,
    ) -> RagasEvaluationTaskResult:
        try:
            start = perf_counter()
            score = await task.metric.single_turn_ascore(task.sample)
            time = perf_counter() - start

            return RagasEvaluationTaskResult(score=score, error=None, time=time)

        except Exception as e:
            time = perf_counter() - start

            return RagasEvaluationTaskResult(score=None, error=str(e), time=time)

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
