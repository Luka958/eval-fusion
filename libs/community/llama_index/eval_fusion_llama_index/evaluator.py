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
from llama_index.core.evaluation import SemanticSimilarityEvaluator

from .em import LlamaIndexProxyEM
from .llm import LlamaIndexProxyLLM
from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    LlamaIndexMetric,
    LlamaIndexMetricType,
    LlamaIndexMetricUnion,
)


class LlamaIndexEvaluationTask(NamedTuple):
    input_id: int
    input: EvaluationInput
    evaluator_id: int
    evaluator: LlamaIndexMetricUnion


class LlamaIndexEvaluationTaskResult(NamedTuple):
    score: float | None
    reason: str | None
    error: str | None
    time: float


class LlamaIndexEvaluator(EvalFusionBaseEvaluator):
    def __init__(
        self, llm_settings: EvalFusionLLMSettings, em_settings: EvalFusionEMSettings
    ):
        self._llm = LlamaIndexProxyLLM(llm_settings)
        self._em = LlamaIndexProxyEM(em_settings)

    def __enter__(self) -> LlamaIndexEvaluator:
        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[LlamaIndexMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        evaluators = [
            metric_type(self._em)
            if metric_type == SemanticSimilarityEvaluator
            else metric_type(self._llm)
            for metric_type in metric_types
        ]

        outputs: list[EvaluationOutput] = []

        for i, input in enumerate(inputs):
            output_entries: list[EvaluationOutputEntry] = []

            for evaluator in evaluators:
                metric_name = evaluator.__class__.__name__.lower().removesuffix(
                    'evaluator'
                )

                try:
                    start = perf_counter()
                    evaluation_result = evaluator.evaluate(
                        query=input.input,
                        response=input.output,
                        contexts=input.relevant_chunks,
                    )
                    time = perf_counter() - start

                    score = evaluation_result.score
                    reason = evaluation_result.feedback

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
        metrics: list[LlamaIndexMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        evaluators = [
            metric_type(self._em)
            if metric_type == SemanticSimilarityEvaluator
            else metric_type(self._llm)
            for metric_type in metric_types
        ]

        metric_type_to_tasks: dict[
            LlamaIndexMetricType, list[LlamaIndexEvaluationTask]
        ] = {}

        for i, input in enumerate(inputs):
            for j, evaluator in enumerate(evaluators):
                task = LlamaIndexEvaluationTask(
                    input_id=i,
                    input=input,
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
                ids_to_entry[(i, j)] = EvaluationOutputEntry(
                    metric_name=evaluator.__class__.__name__.lower().removesuffix(
                        'evaluator'
                    ),
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
        task: LlamaIndexEvaluationTask,
    ) -> LlamaIndexEvaluationTaskResult:
        try:
            start = perf_counter()
            evaluation_result = await task.evaluator.aevaluate(
                query=task.input.input,
                response=task.input.output,
                contexts=task.input.relevant_chunks,
            )
            time = perf_counter() - start

            return LlamaIndexEvaluationTaskResult(
                score=evaluation_result.score,
                reason=evaluation_result.feedback,
                error=None,
                time=time,
            )

        except Exception as e:
            time = perf_counter() - start

            return LlamaIndexEvaluationTaskResult(
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
        self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
