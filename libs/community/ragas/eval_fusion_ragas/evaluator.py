from __future__ import annotations

import asyncio

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
from eval_fusion_core.models.settings import EvalFusionEMSettings, EvalFusionLLMSettings
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy

from .em import RagasProxyEM
from .llm import RagasProxyLLM
from .metrics import FEATURE_TO_METRICS, METRIC_TO_TYPE, RagasMetric


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
        metrics: list[RagasMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_instances = []

        for metric in metrics:
            metric_type = METRIC_TO_TYPE[metric]
            metric_instance = (
                metric_type(llm=self._llm, embeddings=self._em)
                if metric_type is ResponseRelevancy
                else metric_type(llm=self._llm)
            )
            metric_instances.append(metric_instance)

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

        # group (input_idx, metric_idx, metric, sample) by metric class
        metric_type_to_tasks: dict[
            type, list[tuple[int, int, RagasMetric, SingleTurnSample]]
        ] = {}

        for i, sample in enumerate(single_turn_samples):
            for j, metric in enumerate(metric_instances):
                metric_type_to_tasks.setdefault(type(metric), []).append(
                    (i, j, metric, sample)
                )

        results: dict[tuple[int, int], EvaluationOutputEntry] = {}

        for _, tasks in metric_type_to_tasks.items():
            coros = [
                self._score_sample(metric, sample) for (_, _, metric, sample) in tasks
            ]
            batch = await asyncio.gather(*coros)

            for (i, j, metric, _), (score, reason, error, elapsed) in zip(tasks, batch):
                results[(i, j)] = EvaluationOutputEntry(
                    metric_name=metric.name,
                    score=score,
                    reason=reason,
                    error=error,
                    time=elapsed,
                )

        # sort outputs in original order
        outputs: list[EvaluationOutput] = []

        for i, x in enumerate(inputs):
            entries = [results[(i, j)] for j in range(len(metric_instances))]
            outputs.append(EvaluationOutput(input_id=x.id, output_entries=entries))

        return outputs

    async def _score_sample(
        self,
        metric: RagasMetric,
        sample: SingleTurnSample,
    ) -> tuple[float | None, str | None, str | None, float | None]:
        """
        Helper to run a single metric.single_turn_score(sample),
        timing it and catching any exception.
        Returns (score, reason, error, elapsed_time).
        """
        try:
            start = perf_counter()
            score = metric.single_turn_score(sample)
            elapsed = perf_counter() - start
            return score, None, None, elapsed
        except Exception as e:
            return None, None, str(e), None

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
