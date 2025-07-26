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
from ragchecker import RAGChecker, RAGResult, RAGResults
from ragchecker.container import RetrievedDoc
from ragchecker.metrics import (
    METRIC_GROUP_MAP,
    generator_metrics,
    overall_metrics,
    retriever_metrics,
)

from .llm import RagCheckerProxyLLM
from .metrics import FEATURE_TO_METRICS, METRIC_TO_TYPE, RagCheckerMetric


class RagCheckerEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = RagCheckerProxyLLM(settings)

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

        evaluator = RAGChecker(
            custom_llm_api_func=self._llm.custom_llm_api_func,
            batch_size_extractor=1,
            batch_size_checker=1,
        )

        outputs: list[EvaluationOutput] = []

        for i, rag_results in enumerate(jagged_rag_results):
            output_entries: list[EvaluationOutputEntry] = []

            for metric in metric_instances:
                metric_name = metric

                try:
                    start = perf_counter()
                    items = evaluator.evaluate(rag_results, metric_instances)
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

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = self._llm.get_token_usage()
