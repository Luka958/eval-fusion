from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings

from .llm import DeepEvalProxyLLM


class DeepEvalEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.llm: DeepEvalProxyLLM = DeepEvalProxyLLM(settings)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        answer_relevancy = AnswerRelevancyMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
            _show_indicator=False,
        )
        contextual_precision = ContextualPrecisionMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
            _show_indicator=False,
        )
        contextual_recall = ContextualRecallMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
            _show_indicator=False,
        )
        contextual_relevancy = ContextualRelevancyMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
            _show_indicator=False,
        )
        faithfulness = FaithfulnessMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
            _show_indicator=False,
        )

        metrics: list[BaseMetric] = [
            answer_relevancy,
            contextual_precision,
            contextual_recall,
            contextual_relevancy,
            faithfulness,
        ]
        metrics = metrics[:1]  # TODO remove

        test_cases = [
            LLMTestCase(
                input=x.input,
                actual_output=x.output,
                expected_output=x.ground_truth,
                context=None,
                retrieval_context=x.relevant_chunks,
                tools_called=None,
                expected_tools=None,
            )
            for x in inputs
        ]

        return [
            EvaluationOutput(
                input_id=inputs[i].id,
                output_entries=[
                    EvaluationOutputEntry(
                        metric_name=metric.__name__,
                        score=metric.measure(test_case),
                        reason=metric.reason,
                    )
                    for metric in metrics
                ],
            )
            for i, test_case in enumerate(test_cases)
        ]
