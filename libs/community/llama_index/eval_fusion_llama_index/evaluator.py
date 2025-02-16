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
from eval_fusion_core.models.settings import EvalFusionEMSettings, EvalFusionLLMSettings
from llama_index.core.evaluation import SemanticSimilarityEvaluator

from .em import LlamaIndexProxyEM
from .llm import LlamaIndexProxyLLM
from .metrics import FEATURE_TO_METRICS, METRIC_TO_TYPE, LlamaIndexMetric


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
                metric_name = ...

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

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
