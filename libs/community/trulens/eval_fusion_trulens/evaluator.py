from types import TracebackType

from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from trulens.apps.virtual import TruVirtual, VirtualApp, VirtualRecord
from trulens.core import Feedback, FeedbackMode, Select, TruSession
from trulens.core.schema.feedback import FeedbackResultStatus

from .constants import APP_ID
from .llm import TruLensProxyLLM
from .metrics import TAG_TO_METRIC_TYPES, TruLensMetric


class TruLensEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = TruLensProxyLLM(settings)

    def __enter__(self) -> 'TruLensEvaluator':
        self._session = TruSession()

        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[type[TruLensMetric]],
    ) -> list[EvaluationOutput]:
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

        if type(TruLensMetric.CONTEXT_RELEVANCE) in metric_types:
            feedbacks.append(
                Feedback(
                    self._llm.context_relevance_with_cot_reasons,
                    name=TruLensMetric.CONTEXT_RELEVANCE.value,
                )
                .on(Select.RecordInput)
                .on(context_selector)
            )

        if type(TruLensMetric.GROUNDEDNESS) in metric_types:
            feedbacks.append(
                Feedback(
                    self._llm.groundedness_measure_with_cot_reasons,
                    name=TruLensMetric.GROUNDEDNESS.value,
                )
                .on(context_selector.collect())
                .on(output_selector)
            )

        if type(TruLensMetric.ANSWER_RELEVANCE) in metric_types:
            feedbacks.append(
                Feedback(
                    self._llm.relevance_with_cot_reasons,
                    name=TruLensMetric.ANSWER_RELEVANCE.value,
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

            for future in record.feedback_results:
                try:
                    feedback_result = future.result()
                    metric_name = feedback_result.name
                    score = feedback_result.result
                    reason = str(feedback_result.calls[0].meta['reason'])

                    if feedback_result.status == FeedbackResultStatus.DONE:
                        output_entries.append(
                            EvaluationOutputEntry(
                                metric_name=metric_name,
                                score=score,
                                reason=reason,
                            )
                        )

                    else:
                        output_entries.append(
                            EvaluationOutputEntry(
                                metric_name=metric_name,
                                error=e,
                            )
                        )

                except Exception as e:
                    output_entries.append(
                        EvaluationOutputEntry(
                            metric_name=metric_name,  # TODO not available yet
                            error=e,
                        )
                    )

            outputs.append(
                EvaluationOutput(
                    input_id=inputs[i].id,
                    output_entries=output_entries,
                )
            )

        return outputs

    def evaluate_by_tag(
        self,
        inputs: list[EvaluationInput],
        tag: MetricTag,
    ) -> list[EvaluationOutput]:
        if tag is not None:
            metric_types = TAG_TO_METRIC_TYPES[tag]

        return self.evaluate(inputs, metric_types)

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._session.delete_app(APP_ID)
