from types import TracebackType

from eval_fusion_core.base import (
    EvalFusionBaseEvaluator,
    EvalFusionBaseLLM,
)
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from mlflow import (
    create_experiment,
    evaluate,
    register_model,
    set_experiment,
    start_run,
)
from mlflow.metrics.genai import (
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
)
from mlflow.models.evaluation import EvaluationMetric, EvaluationResult
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import log_model
from pandas import DataFrame
from requests.exceptions import ConnectionError

from .llm import MlFlowProxyLLM
from .utils.processes import close_process, open_process


class MlFlowEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.llm: MlFlowProxyLLM = MlFlowProxyLLM(settings)

    def __enter__(self) -> 'MlFlowEvaluator':
        EXPERIMENT_NAME = 'eval_fusion_experiment'
        self.experiment_id = create_experiment(EXPERIMENT_NAME)
        set_experiment(EXPERIMENT_NAME)

        signature = infer_signature(
            model_input=['What is MLflow?'],
            model_output=[
                'MLflow is a platform for managing machine learning workflows.'
            ],
        )

        ARTIFACT_PATH = 'eval_fusion_llm'

        with start_run():
            model_info = log_model(
                artifact_path=ARTIFACT_PATH, python_model=self.llm, signature=signature
            )

        MODEL_NAME = 'custom_llm'

        model_version = register_model(model_uri=model_info.model_uri, name=MODEL_NAME)

        self.models_process = open_process(
            [
                'mlflow',
                'models',
                'serve',
                '--model-uri',
                f'models:/custom_llm/{model_version.version}',
                '--host',
                '127.0.0.1',
                '--port',
                '5000',
                '--env-manager',
                'local',
            ]
        )

        self.deployments_process = open_process(
            [
                'mlflow',
                'deployments',
                'start-server',
                '--config-path',
                'config.yaml',
                '--host',
                '127.0.0.1',
                '--port',
                '5001',
            ]
        )

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        model = self.llm.get_model()

        metrics: list[EvaluationMetric] = [
            faithfulness(model),
            answer_correctness(model),
            answer_relevance(model),
            answer_similarity(model),
            relevance(model),
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

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        close_process(self.models_process.pid)
        close_process(self.deployments_process.pid)
