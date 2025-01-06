from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import EvaluationInput, EvaluationOutput

from .llm import DeepEvalLLM


class DeepEvalEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.llm: DeepEvalLLM = DeepEvalLLM(llm_delegate=llm)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        answer_relevancy = AnswerRelevancyMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        contextual_precision = ContextualPrecisionMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        contextual_recall = ContextualRecallMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        contextual_relevancy = ContextualRelevancyMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        faithfulness = FaithfulnessMetric(
            threshold=0.5,
            model=self.llm,
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
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

        outputs: list[EvaluationOutput] = []

        for test_case in test_cases:
            for metric in metrics:
                metric.measure(test_case)
                output = EvaluationOutput(score=metric.score)
                outputs.append(output)

        return outputs
