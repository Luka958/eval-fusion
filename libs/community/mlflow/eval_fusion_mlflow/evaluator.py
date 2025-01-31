from time import perf_counter
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
from mlflow.data.pandas_dataset import from_pandas
from mlflow.deployments import set_deployments_target
from mlflow.models.evaluation.evaluators.default import DefaultEvaluator
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import log_model
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from pandas import DataFrame

from .constants import *
from .llm import MlFlowProxyLLM
from .metrics import TAG_TO_METRIC_TYPES, MlFlowMetric
from .utils.connections import check_health
from .utils.processes import close_process, open_process, run_process


class MlFlowEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        self._llm = MlFlowProxyLLM(settings)

    def __enter__(self) -> 'MlFlowEvaluator':
        self._experiment_id = create_experiment(EXPERIMENT_NAME)
        set_experiment(self._experiment_id)

        signature = ModelSignature(
            inputs=Schema([ColSpec(type=DataType.string, required=True)]),
            outputs=Schema([ColSpec(type=DataType.string, required=True)]),
            params=ParamSchema(
                [
                    ParamSpec(name='temperature', dtype=DataType.float, default=0.0),
                    ParamSpec(name='n', dtype=DataType.integer, default=1),
                    ParamSpec(name='max_tokens', dtype=DataType.integer, default=1024),
                    ParamSpec(name='top_p', dtype=DataType.float, default=1.0),
                ]
            ),
        )

        with start_run():
            model_info = log_model(
                artifact_path=ARTIFACT_PATH, python_model=self._llm, signature=signature
            )

        model_version = register_model(model_uri=model_info.model_uri, name=MODEL_NAME)

        self._models_process = open_process(
            [
                'mlflow',
                'models',
                'serve',
                '--model-uri',
                f'models:/{MODEL_NAME}/{model_version.version}',
                '--host',
                MODELS_HOST,
                '--port',
                str(MODELS_PORT),
                '--env-manager',
                MODELS_ENV_MANAGER,
            ]
        )
        self._deployments_process = open_process(
            [
                'mlflow',
                'deployments',
                'start-server',
                '--config-path',
                DEPLOYMENTS_CONFIG_PATH,
                '--host',
                DEPLOYMENTS_HOST,
                '--port',
                str(DEPLOYMENTS_PORT),
            ]
        )

        check_health(MODELS_HOST, MODELS_PORT)
        check_health(DEPLOYMENTS_HOST, DEPLOYMENTS_PORT)

        set_deployments_target(f'http://{DEPLOYMENTS_HOST}:{DEPLOYMENTS_PORT}')

        return self

    def evaluate(
        self,
        inputs: list[EvaluationInput],
        metric_types: list[MlFlowMetric],
    ) -> list[EvaluationOutput]:
        metrics = [metric_type(model=MODEL) for metric_type in metric_types]

        data_frames = [
            DataFrame(
                [
                    {
                        'inputs': [x.input],
                        'context': ['\n\n'.join(x.relevant_chunks)],
                        'predictions': [x.output],
                        'targets': [x.ground_truth],
                    }
                ]
            )
            for x in inputs
        ]
        pandas_datasets = list(
            map(
                lambda x: from_pandas(x, predictions='predictions', targets='targets'),
                data_frames,
            )
        )
        evaluation_datasets = [x.to_evaluation_dataset() for x in pandas_datasets]

        default_evaluator = DefaultEvaluator()

        outputs: list[EvaluationOutput] = []

        with start_run() as run:
            for i, evaluation_dataset in enumerate(evaluation_datasets):
                output_entries: list[EvaluationOutputEntry] = []

                for j, metric in enumerate(metrics):
                    metric_name = metric_types[j].__name__

                    try:
                        start = perf_counter()
                        result = default_evaluator.evaluate(
                            run_id=run.info.run_id,
                            dataset=evaluation_dataset,
                            model=None,
                            model_type=None,
                            extra_metrics=[metric],
                            evaluator_config={},
                        )
                        time = perf_counter() - start

                        table = result.tables['eval_results_table']
                        row = i * len(metrics) + j
                        version = str(
                            result.tables['genai_custom_metrics']['version'][row]
                        )
                        score = float(table[f'{metric_name}/{version}/score'].iloc[row])
                        normalized_score = (score - 1) / 4
                        reason = str(
                            table[f'{metric_name}/{version}/justification'].iloc[row]
                        )

                        output_entries.append(
                            EvaluationOutputEntry(
                                metric_name=metric_name,
                                score=normalized_score,
                                reason=reason,
                                error=None,
                                time=time,
                            )
                        )

                    except Exception as e:
                        output_entries.append(
                            EvaluationOutputEntry(
                                metric_name=metric_name,
                                score=None,
                                reason=None,
                                error=str(e),
                                time=None,
                            )
                        )

                outputs.append(
                    EvaluationOutput(
                        input_id=inputs[i].id,
                        output_entries=output_entries,
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
        self.token_usage = self._llm.get_token_usage()

        close_process(self._models_process.pid)
        close_process(self._deployments_process.pid)

        delete_experiment(self._experiment_id)

        client = MlflowClient()
        client.delete_registered_model(MODEL_NAME)

        run_process(['mlflow', 'gc', '--experiment-ids', self._experiment_id])
