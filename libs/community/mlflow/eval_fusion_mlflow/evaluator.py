from types import TracebackType

from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from mlflow import (
    create_experiment,
    delete_experiment,
    register_model,
    set_experiment,
    start_run,
)
from mlflow.data.evaluation_dataset import convert_data_to_mlflow_dataset
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai import (
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
)
from mlflow.models.evaluation import EvaluationMetric
from mlflow.models.evaluation.evaluators.default import DefaultEvaluator
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import log_model
from mlflow.tracking import MlflowClient
from pandas import DataFrame

from .constants import *
from .llm import MlFlowProxyLLM
from .metrics import TAG_TO_METRIC_TYPES
from .utils.connections import check_health
from .utils.processes import close_process, open_process


class MlFlowEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.llm: MlFlowProxyLLM = MlFlowProxyLLM(settings)

    def __enter__(self) -> 'MlFlowEvaluator':
        self.experiment_id = create_experiment(EXPERIMENT_NAME)
        set_experiment(EXPERIMENT_NAME)

        signature = infer_signature(
            model_input=['What is MLflow?'],
            model_output=[
                'MLflow is a platform for managing machine learning workflows.'
            ],
        )

        with start_run():
            model_info = log_model(
                artifact_path=ARTIFACT_PATH, python_model=self.llm, signature=signature
            )

        model_version = register_model(model_uri=model_info.model_uri, name=MODEL_NAME)

        self.models_process = open_process(
            [
                'mlflow',
                'models',
                'serve',
                '--model-uri',
                f'models:/{MODEL_NAME}/{model_version.version}',
                '--host',
                MODELS_HOST,
                '--port',
                MODELS_PORT,
                '--env-manager',
                MODELS_ENV_MANAGER,
            ]
        )

        self.deployments_process = open_process(
            [
                'mlflow',
                'deployments',
                'start-server',
                '--config-path',
                DEPLOYMENTS_CONFIG_PATH,
                '--host',
                DEPLOYMENTS_HOST,
                '--port',
                DEPLOYMENTS_PORT,
            ]
        )

        check_health(MODELS_HOST, MODELS_PORT)
        check_health(DEPLOYMENTS_HOST, DEPLOYMENTS_PORT)

        set_deployments_target(f'http://{DEPLOYMENTS_HOST}:{DEPLOYMENTS_PORT}')

        return self

    def evaluate(
        self, inputs: list[EvaluationInput], metrics: list
    ) -> list[EvaluationOutput]:
        # TODO organize metrics

        ENDPOINT_NAME = 'chat'

        model = f'endpoints:/{ENDPOINT_NAME}'

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
                    'predictions': x.ground_truth,
                }
                for x in inputs
            ]
        )

        pandas_dataset: PandasDataset = convert_data_to_mlflow_dataset(
            data_frame, predictions='predictions'
        )
        evaluation_dataset = pandas_dataset.to_evaluation_dataset()
        default_evaluator = DefaultEvaluator()

        evaluation_outputs: list[EvaluationOutput] = []

        with start_run() as run:
            for i, (_, series) in enumerate(data_frame.iterrows()):
                evaluation_output_entires: list[EvaluationOutputEntry] = []

                for metric in metrics:
                    evaluation_result = default_evaluator.evaluate(
                        run_id=run.info.run_id,
                        dataset=evaluation_dataset,
                        model=None,
                        model_type=None,
                        extra_metrics=[metric],
                        evaluator_config={},
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
        close_process(self.models_process.pid)
        close_process(self.deployments_process.pid)

        delete_experiment(self.experiment_id)

        client = MlflowClient()
        client.delete_registered_model(MODEL_NAME)
