from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from eval_fusion_core.models import EvaluationInput, EvaluationOutput, Evaluator


class DeepEvalEvaluator(Evaluator):
    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO adapt models

        answer_relevancy = AnswerRelevancyMetric(
            threshold=0.5,
            model=None,  # TODO
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        contextual_precision = ContextualPrecisionMetric(
            threshold=0.5,
            model=None,  # TODO
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        contextual_recall = ContextualRecallMetric(
            threshold=0.5,
            model=None,  # TODO
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        contextual_relevancy = ContextualRelevancyMetric(
            threshold=0.5,
            model=None,  # TODO
            include_reason=False,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        faithfulness = FaithfulnessMetric(
            threshold=0.5,
            model=None,  # TODO
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
