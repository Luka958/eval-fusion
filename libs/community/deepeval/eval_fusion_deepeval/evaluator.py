from deepeval.metrics import AnswerRelevancyMetric, BaseMetric
from deepeval.test_case import LLMTestCase
from eval_fusion_core.models import EvaluationInput, EvaluationOutput, Evaluator


class DeepEvalEvaluator(Evaluator):
    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO adapt inputs to DeepEval
        # TODO adapt outputs from DeepEval
        # TODO adapt models

        metrics: list[BaseMetric] = [AnswerRelevancyMetric(threshold=0.5)]
        test_cases = [LLMTestCase(input='...', actual_output='...')]
        outputs = []

        for test_case in test_cases:
            for metric in metrics:
                metric.measure(test_case)  # TODO score, reason, breakdown
                outputs.append(metric.score)

        return outputs
