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
from ragchecker import RAGChecker, RAGResult, RAGResults
from ragchecker.container import RetrievedDoc
from ragchecker.metrics import (
    METRIC_GROUP_MAP,
    generator_metrics,
    overall_metrics,
    retriever_metrics,
)

from .llm import RagCheckerProxyLLM
from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    RagCheckerMetric,
    RagCheckerMetricType,
)


class RagCheckerEvaluationTask(NamedTuple):
    rag_results_id: int
    rag_results: RAGResults
    metric_id: int
    metric: RagCheckerMetricType


class RagCheckerEvaluationTaskResult(NamedTuple):
    score: float | None
    error: str | None
    time: float


class RagCheckerEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = RagCheckerProxyLLM(settings)
        self._rag_checker = RAGChecker(
            custom_llm_api_func=self._llm.custom_llm_api_func,
            batch_size_extractor=1,
            batch_size_checker=1,
        )

    def __enter__(self) -> RagCheckerEvaluator:
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[RagCheckerMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = metric_types

        jagged_rag_results = [
            RAGResults(
                results=[
                    RAGResult(
                        query_id=str(i),
                        query=x.input,
                        gt_answer=x.ground_truth,
                        response=x.output,
                        retrieved_context=[
                            RetrievedDoc(doc_id=str(j), text=y)
                            for j, y in enumerate(x.relevant_chunks)
                        ],
                    )
                ]
            )
            for i, x in enumerate(inputs)
        ]

        outputs: list[EvaluationOutput] = []

        for i, rag_results in enumerate(jagged_rag_results):
            output_entries: list[EvaluationOutputEntry] = []

            for metric in metric_instances:
                metric_name = metric

                try:
                    start = perf_counter()
                    items = self._rag_checker.evaluate(rag_results, [metric])
                    time = perf_counter() - start

                    if metric in METRIC_GROUP_MAP[overall_metrics]:
                        metric_group = items.get(overall_metrics)

                    elif metric in METRIC_GROUP_MAP[retriever_metrics]:
                        metric_group = items.get(retriever_metrics)

                    elif metric in METRIC_GROUP_MAP[generator_metrics]:
                        metric_group = items.get(generator_metrics)

                    else:
                        raise EvalFusionException(
                            f'Metric {metric} does not belong to any group.'
                        )

                    score = float(metric_group.get(metric))

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
        metrics: list[RagCheckerMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = metric_types

        jagged_rag_results = [
            RAGResults(
                results=[
                    RAGResult(
                        query_id=str(i),
                        query=x.input,
                        gt_answer=x.ground_truth,
                        response=x.output,
                        retrieved_context=[
                            RetrievedDoc(doc_id=str(j), text=y)
                            for j, y in enumerate(x.relevant_chunks)
                        ],
                    )
                ]
            )
            for i, x in enumerate(inputs)
        ]

        metric_type_to_tasks: dict[
            RagCheckerMetricType, list[RagCheckerEvaluationTask]
        ] = {}

        for i, rag_results in enumerate(jagged_rag_results):
            for j, metric in enumerate(metric_instances):
                task = RagCheckerEvaluationTask(
                    rag_results_id=i,
                    rag_results=rag_results,
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
                    metric_name=metric,
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
        task: RagCheckerEvaluationTask,
    ) -> RagCheckerEvaluationTaskResult:
        try:
            start = perf_counter()
            loop = asyncio.get_event_loop()
            items = await loop.run_in_executor(
                None,
                self._rag_checker.evaluate,
                task.rag_results,
                [task.metric],
            )
            time = perf_counter() - start

            metric = task.metric

            if metric in METRIC_GROUP_MAP[overall_metrics]:
                metric_group = items.get(overall_metrics)

            elif metric in METRIC_GROUP_MAP[retriever_metrics]:
                metric_group = items.get(retriever_metrics)

            elif metric in METRIC_GROUP_MAP[generator_metrics]:
                metric_group = items.get(generator_metrics)

            else:
                raise EvalFusionException(
                    f'Metric {metric} does not belong to any group.'
                )

            score = float(metric_group.get(metric))

            return RagCheckerEvaluationTaskResult(score=score, error=None, time=time)

        except Exception as e:
            time = perf_counter() - start

            return RagCheckerEvaluationTaskResult(score=None, error=str(e), time=time)

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = self._llm.get_token_usage()
