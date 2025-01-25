from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from trulens.apps.virtual import TruVirtual, VirtualApp
from trulens.core import Feedback

from .llm import TruLensProxyLLM


class TruLensEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.llm: TruLensProxyLLM = TruLensProxyLLM(llm_delegate=llm)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        virtual_app = VirtualApp()
        context = VirtualApp.select_context()

        f_context_relevance = (
            Feedback(
                self.llm.context_relevance_with_cot_reasons, name='Context Relevance'
            )
            .on_input()
            .on(context)
        )
        f_groundedness = (
            Feedback(
                self.llm.groundedness_measure_with_cot_reasons, name='Groundedness'
            )
            .on(context.collect())
            .on_output()
        )
        f_qa_relevance = Feedback(
            self.llm.relevance_with_cot_reasons, name='Answer Relevance'
        ).on_input_output()

        virtual_recorder = TruVirtual(
            app_name='RAG',
            app_version='simple',
            app=virtual_app,
            feedbacks=[f_context_relevance, f_groundedness, f_qa_relevance],
        )
