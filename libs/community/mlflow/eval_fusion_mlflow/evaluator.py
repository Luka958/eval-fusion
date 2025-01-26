from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from mlflow import evaluate
from mlflow.metrics.genai import (
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
)
from mlflow.models.evaluation import EvaluationMetric, EvaluationResult
from pandas import DataFrame

from .llm import MlFlowProxyLLM


class MlFlowEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, llm: EvalFusionBaseLLM):
        self.llm: MlFlowProxyLLM = MlFlowProxyLLM(llm_delegate=llm)

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        model = self.llm.get_model()

        metrics: list[EvaluationMetric] = [
            faithfulness(model=model),
            answer_correctness(model=model),
            answer_relevance(model=model),
            answer_similarity(model=model),
            relevance(model=model),
        ]

        data_frame = DataFrame(
            [
                {
                    'inputs': x.input,
                    'context': x.relevant_chunks,
                    'answers': x.output,
                    'targets': x.ground_truth,
                }
                for x in inputs
            ]
        )

        evaluation_outputs: list[EvaluationOutput] = []

        for i, (_, series) in enumerate(data_frame.iterrows()):
            evaluation_output_entires: list[EvaluationOutputEntry] = []

            for metric in metrics:
                evaluation_result: EvaluationResult = evaluate(
                    model=None,
                    data=series.to_frame().T,
                    model_type=None,
                    evaluators=None,
                    predictions='targets',
                    extra_metrics=[metric],
                )
                table = evaluation_result.tables['eval_results_table']

                evaluation_output_entires.append(
                    EvaluationOutputEntry(
                        metric_name=metric.__name__,
                        score=table[f'{metric.__name__}/v1/score'],
                        reason=table[f'{metric.__name__}/v1/justification'],
                    )
                )

            evaluation_outputs.append(
                EvaluationOutput(
                    input_id=inputs[i].id,
                    output_entries=evaluation_output_entires,
                )
            )

        return evaluation_outputs
