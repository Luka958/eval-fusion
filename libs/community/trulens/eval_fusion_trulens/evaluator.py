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
from trulens.apps.virtual import TruVirtual, VirtualApp, VirtualRecord
from trulens.core import Feedback, FeedbackMode, Select, TruSession
from trulens.core.database.connector.default import DefaultDBConnector
from trulens.core.experimental import Feature
from trulens.core.feedback import Endpoint
from trulens.core.schema.feedback import FeedbackResultStatus

from .constants import APP_ID
from .llm import TruLensProxyLLM
from .metrics import (
    FEATURE_TO_METRICS,
    METRIC_TO_TYPE,
    ContextRelevance,
    Groundedness,
    Relevance,
    TruLensMetric,
)


class TruLensEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = TruLensProxyLLM(
            tru_class_info=TruLensProxyLLM,
            endpoint=Endpoint(name='eval_fusion_endpoint'),
            model_engine='',
            settings=settings,
        )

    def __enter__(self) -> TruLensEvaluator:
        self._session = TruSession(
            connector=DefaultDBConnector(database_url='sqlite:///:memory:')
        )
        self._session.experimental_disable_feature(Feature.OTEL_TRACING)

        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[TruLensMetric] | None = None,
        feature: Feature | None = None,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and tag cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))

        retriever = Select.RecordCalls.retriever
        synthesizer = Select.RecordCalls.synthesizer

        virtual_records = [
            VirtualRecord(
                main_input=x.input,
                main_output=x.output,
                calls={
                    retriever.get_context: dict(
                        args=[x.input], rets=['\n\n'.join(x.relevant_chunks)]
                    ),
                    synthesizer.generate: dict(args=[x.input], rets=[x.output]),
                },
            )
            for x in inputs
        ]

        context_selector = retriever.get_context.rets[:]
        output_selector = synthesizer.generate.rets[:]

        feedbacks: list[Feedback] = []

        if ContextRelevance in metric_types:
            feedbacks.append(
                Feedback(
                    self._llm.context_relevance_with_cot_reasons,
                    name='context_relevance',
                )
                .on(Select.RecordInput)
                .on(context_selector)
            )

        if Groundedness in metric_types:
            feedbacks.append(
                Feedback(
                    self._llm.groundedness_measure_with_cot_reasons,
                    name='groundedness',
                )
                .on(context_selector.collect())
                .on(output_selector)
            )

        if Relevance in metric_types:
            feedbacks.append(
                Feedback(
                    self._llm.relevance_with_cot_reasons,
                    name='relevance',
                ).on_input_output()
            )

        tru = TruVirtual(
            app=VirtualApp(),
            app_id=APP_ID,
            feedbacks=feedbacks,
        )

        outputs: list[EvaluationOutput] = []

        for i, record in enumerate(virtual_records):
            output_entries: list[EvaluationOutputEntry] = []

            tru.add_record(record, FeedbackMode.WITH_APP_THREAD)

            for j, future in enumerate(record.feedback_results):
                metric_name = feedbacks[j].name

                try:
                    start = perf_counter()
                    feedback_result = future.result()
                    time = perf_counter() - start

                    score = feedback_result.result
                    feedback_call = feedback_result.calls[0]
                    reason = feedback_call.meta.get(
                        'reason', feedback_call.meta.get('reasons')
                    )

                    if feedback_result.status == FeedbackResultStatus.FAILED:
                        output_entries.append(
                            EvaluationOutputEntry(
                                metric_name=metric_name,
                                score=None,
                                reason=None,
                                error=feedback_result.error,
                                time=None,
                            )
                        )

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
        self.token_usage = self._llm.get_token_usage()

        self._session.delete_app(APP_ID)
