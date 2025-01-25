from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from trulens.apps.virtual import TruVirtual, VirtualApp, VirtualRecord
from trulens.core import Feedback, FeedbackMode, Select, TruSession
from trulens.core.schema.feedback import FeedbackResultStatus

from .llm import TruLensProxyLLM


class TruLensEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.llm: TruLensProxyLLM = TruLensProxyLLM(llm_delegate=llm)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics
        retriever = Select.RecordCalls.retriever
        synthesizer = Select.RecordCalls.synthesizer

        virtual_records = [
            VirtualRecord(
                main_input=x.input,
                main_output=x.output,
                calls={
                    retriever.get_context: dict(args=[x.input], rets=x.relevant_chunks),
                    synthesizer.generate: dict(args=[x.input], rets=[x.output]),
                },
            )
            for x in inputs
        ]

        context_selector = retriever.get_context.rets[:]
        output_selector = synthesizer.generate.rets[:]

        context_relevance_feedback = (
            Feedback(
                self.llm.context_relevance_with_cot_reasons, name='context_relevance'
            )
            .on(Select.RecordInput)
            .on(context_selector)
        )
        groundedness_feedback = (
            Feedback(
                self.llm.groundedness_measure_with_cot_reasons, name='groundedness'
            )
            .on(context_selector.collect())
            .on(output_selector)
        )
        answer_relevance_feedback = Feedback(
            self.llm.relevance_with_cot_reasons, name='answer_relevance'
        ).on_input_output()

        feedbacks: list[Feedback] = [
            context_relevance_feedback,
            groundedness_feedback,
            answer_relevance_feedback,
        ]

        session = TruSession()
        app_id = 'my_app_id'
        tru = TruVirtual(
            app=VirtualApp(),
            app_id=app_id,
            feedbacks=feedbacks,
        )

        outputs: list[EvaluationOutput] = []

        for i, record in enumerate(virtual_records):
            tru.add_record(record, FeedbackMode.WITH_APP_THREAD)

            # TODO multiple reasons
            output = EvaluationOutput(
                input_id=inputs[i].id,
                output_entries=[
                    EvaluationOutputEntry(
                        metric_name=feedback_result.name,
                        score=feedback_result.result,
                        reason=None,  # [call.meta['reason'] for call in feedback_result.calls]
                    )
                    for feedback_result in [
                        future.result() for future in record.feedback_results
                    ]
                    if feedback_result.status == FeedbackResultStatus.DONE
                ],
            )
            outputs.append(output)

        session.delete_app(app_id)

        return outputs
