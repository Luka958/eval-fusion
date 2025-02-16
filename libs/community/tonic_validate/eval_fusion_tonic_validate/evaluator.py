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
from tonic_validate import BenchmarkItem, LLMResponse

from .llm import TonicValidateProxyLLM
from .metrics import FEATURE_TO_METRICS, METRIC_TO_TYPE, TonicValidateMetric


class TonicValidateEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm_settings: EvalFusionLLMSettings):
        self._llm = TonicValidateProxyLLM(llm_settings)

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
        metric_instances = [
            metric_type(llm=self._llm, embeddings=self._em)
            if metric_type == ResponseRelevancy
            else metric_type(llm=self._llm)
            for metric_type in metric_types
        ]

        responses = [
            LLMResponse(
                llm_answer='Paris',
                llm_context_list=['Paris is the capital of France.'],
                benchmark_item=BenchmarkItem(question=..., answer=...),
            )
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

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.token_usage = (self._llm.get_token_usage(), self._em.get_token_usage())
