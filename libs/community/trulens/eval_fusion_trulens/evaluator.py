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
        self.llm = TruLensProxyLLM(settings)

    def __enter__(self) -> 'TruLensEvaluator':
        self.session = TruSession()

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
                    self.llm.context_relevance_with_cot_reasons,
                    name=TruLensMetric.CONTEXT_RELEVANCE.value,
                )
                .on(Select.RecordInput)
                .on(context_selector)
            )

        if type(TruLensMetric.GROUNDEDNESS) in metric_types:
            feedbacks.append(
                Feedback(
                    self.llm.groundedness_measure_with_cot_reasons,
                    name=TruLensMetric.GROUNDEDNESS.value,
                )
                .on(context_selector.collect())
                .on(output_selector)
            )

        if type(TruLensMetric.ANSWER_RELEVANCE) in metric_types:
            feedbacks.append(
                Feedback(
                    self.llm.relevance_with_cot_reasons,
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
            tru.add_record(record, FeedbackMode.WITH_APP_THREAD)

            outputs.append(
                EvaluationOutput(
                    input_id=inputs[i].id,
                    output_entries=[
                        EvaluationOutputEntry(
                            metric_name=feedback_result.name,
                            score=feedback_result.result,
                            reason=feedback_result.calls[0].meta['reason'],
                        )
                        for feedback_result in [
                            future.result() for future in record.feedback_results
                        ]
                        if feedback_result.status == FeedbackResultStatus.DONE
                    ],
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
        self.session.delete_app(APP_ID)
