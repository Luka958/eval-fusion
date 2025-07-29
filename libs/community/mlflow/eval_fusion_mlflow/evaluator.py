from __future__ import annotations

import asyncio
import json

from time import perf_counter
from types import TracebackType
from typing import NamedTuple

from eval_fusion_core.base import EvalFusionBaseEvaluator
from eval_fusion_core.enums import Feature
from eval_fusion_core.exceptions import EvalFusionException
from eval_fusion_core.models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationOutputEntry,
    TokenUsage,
)
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from mlflow import (
    create_experiment,
    delete_experiment,
    get_experiment_by_name,
    get_tracking_uri,
    register_model,
    set_experiment,
    start_run,
)
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pandas_dataset import from_pandas
from mlflow.deployments import set_deployments_target
from mlflow.models import EvaluationMetric
from mlflow.models.evaluation.evaluators.default import DefaultEvaluator
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import log_model
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from pandas import DataFrame

from .constants import *
from .metrics import FEATURE_TO_METRICS, METRIC_TO_TYPE, MlFlowMetric
from .utils.connections import check_health
from .utils.processes import close_process, open_process, run_process


class MlFlowEvaluationTask(NamedTuple):
    run_id: str
    row: int

    dataset_id: int
    dataset: EvaluationDataset
    metric_id: int
    metric: EvaluationMetric


class MlFlowEvaluationTaskResult(NamedTuple):
    score: float | None
    reason: str | None
    error: str | None
    time: float


class MlFlowEvaluator(EvalFusionBaseEvaluator):
    def __init__(self, settings: EvalFusionLLMSettings):
        cls = settings.base_type
        fqn = f'{cls.__module__}.{cls.__qualname__}'
        settings.base_type = fqn

        safe_settings = settings.model_copy()
        self._api_key = safe_settings.kwargs.pop('api_key', None)

        with open(LLM_SETTINGS_PATH, 'w') as file:
            json.dump(safe_settings.model_dump(), file)

    def __enter__(self) -> MlFlowEvaluator:
        self._client = MlflowClient()

        experiment = get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            if experiment.lifecycle_stage == 'deleted':
                self._client.restore_experiment(experiment.experiment_id)

            self._experiment_id = experiment.experiment_id

        else:
            self._experiment_id = create_experiment(EXPERIMENT_NAME)

        set_experiment(experiment_id=self._experiment_id)

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
                artifacts={ARTIFACT_KEY_SETTINGS: LLM_SETTINGS_PATH},
                model_config={
                    'api_key': self._api_key,
                    'experiment_id': self._experiment_id,
                }
                if self._api_key is not None
                else None,
                python_model=LLM_PATH,
                signature=signature,
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
                'gateway',
                'start',
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
        metrics: list[MlFlowMetric] | None = None,
        feature: Feature | None = None,
        include_reason: bool = False,
    ) -> list[EvaluationOutput]:
        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [
            metric_type(model=MODEL, max_workers=1) for metric_type in metric_types
        ]

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

                for j, metric in enumerate(metric_instances):
                    metric_name = metric.name
                    row = i * len(metrics) + j

                    try:
                        start = perf_counter()
                        result = default_evaluator.evaluate(
                            run_id=run.info.run_id,
                            dataset=evaluation_dataset,
                            model_type=None,
                            extra_metrics=[metric],
                            evaluator_config={},
                        )
                        time = perf_counter() - start

                        metrics_table = result.tables['genai_custom_metrics']
                        version = metrics_table.loc[
                            metrics_table['name'] == metric_name, 'version'
                        ].iloc[0]

                        results_table = result.tables['eval_results_table']
                        score_series = results_table[f'{metric_name}/{version}/score']
                        score = float(score_series.iloc[row])
                        normalized_score = (score - 1) / 4
                        reason_series = results_table[
                            f'{metric_name}/{version}/justification'
                        ]
                        reason = str(reason_series.iloc[row])

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

    async def a_evaluate(
        self,
        inputs: list[EvaluationInput],
        metrics: list[MlFlowMetric] | None = None,
        feature: Feature | None = None,
        include_reason: bool = False,
    ) -> list[EvaluationOutput]:
        raise NotImplementedError('wip')

        if metrics is None and feature is None:
            raise EvalFusionException('metrics and feature cannot both be None.')

        if feature is not None:
            metrics = FEATURE_TO_METRICS[feature]

        metric_types = list(map(METRIC_TO_TYPE.get, metrics))
        metric_instances = [
            metric_type(model=MODEL, max_workers=1) for metric_type in metric_types
        ]

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

        self._default_evaluator = DefaultEvaluator()

        with start_run() as run:
            metric_type_to_tasks: dict[str, list[MlFlowEvaluationTask]] = {}

            for i, dataset in enumerate(evaluation_datasets):
                for j, metric in enumerate(metric_instances):
                    row = i * len(metrics) + j
                    task = MlFlowEvaluationTask(
                        run_id=run.info.run_id,
                        row=row,
                        dataset_id=i,
                        dataset=dataset,
                        metric_id=j,
                        metric=metric,
                    )
                    metric_type_to_tasks.setdefault(metric.name, []).append(task)

            ids_to_entry: dict[tuple[int, int], EvaluationOutputEntry] = {}

            for _, tasks in metric_type_to_tasks.items():
                coros = [self._run_task(task) for task in tasks]
                batch = await asyncio.gather(*coros)

                for task, result in zip(tasks, batch):
                    _, _, i, _, j, metric = task
                    score, reason, error, time = result
                    ids_to_entry[(i, j)] = EvaluationOutputEntry(
                        metric_name=metric.name,
                        score=score,
                        reason=reason,
                        error=error,
                        time=time,
                    )

            outputs: list[EvaluationOutput] = []

            for i, x in enumerate(inputs):
                entries = [ids_to_entry[(i, j)] for j in range(len(metric_instances))]
                outputs.append(EvaluationOutput(input_id=x.id, output_entries=entries))

            return outputs

    async def _run_task(
        self,
        task: MlFlowEvaluationTask,
    ) -> MlFlowEvaluationTaskResult:
        start = perf_counter()
        try:
            metric_name = task.metric.name
            row = task.row

            loop = asyncio.get_event_loop()

            def _evaluate():
                with start_run(run_id=task.run_id, nested=True):
                    return self._default_evaluator.evaluate(
                        run_id=task.run_id,
                        dataset=task.dataset,
                        model_type=None,
                        extra_metrics=[task.metric],
                        evaluator_config={},
                    )

            result = await loop.run_in_executor(None, _evaluate)
            time = perf_counter() - start

            metrics_table = result.tables['genai_custom_metrics']
            version = metrics_table.loc[
                metrics_table['name'] == metric_name, 'version'
            ].iloc[0]

            results_table = result.tables['eval_results_table']
            score_series = results_table[f'{metric_name}/{version}/score']
            score = float(score_series.iloc[row])
            normalized_score = (score - 1) / 4
            reason_series = results_table[f'{metric_name}/{version}/justification']
            reason = str(reason_series.iloc[row])

            return MlFlowEvaluationTaskResult(
                score=normalized_score,
                reason=reason,
                error=None,
                time=time,
            )

        except Exception as e:
            time = perf_counter() - start

            return MlFlowEvaluationTaskResult(
                score=None,
                reason=None,
                error=str(e),
                time=time,
            )

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string="tags.purpose = 'token-usage'",
            order_by=['attributes.start_time DESC'],
            max_results=1,
        )

        if len(runs) != 1:
            raise ValueError()

        run_id = runs[0].info.run_id

        input_tokens_metrics = self._client.get_metric_history(run_id, 'input_tokens')
        input_tokens = int(input_tokens_metrics[-1].value)
        output_tokens_metrics = self._client.get_metric_history(run_id, 'output_tokens')
        output_tokens = int(output_tokens_metrics[-1].value)
        self.token_usage = TokenUsage(input=input_tokens, output=output_tokens)

        self._client.delete_run(run_id)

        close_process(self._models_process.pid)
        close_process(self._deployments_process.pid)

        delete_experiment(self._experiment_id)
        self._client.delete_registered_model(MODEL_NAME)

        tracking_uri = get_tracking_uri()
        run_process(
            [
                'mlflow',
                'gc',
                '--experiment-ids',
                self._experiment_id,
                '--tracking-uri',
                tracking_uri,
            ]
        )
