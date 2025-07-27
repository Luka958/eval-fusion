import importlib
import json
import os

from eval_fusion_core.exceptions import EvalFusionException
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from mlflow import log_metric, set_tag, start_run
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel, PythonModelContext
from pandas import DataFrame

from eval_fusion_mlflow.constants import ARTIFACT_KEY_SETTINGS


class MlFlowProxyLLM(PythonModel):
    def load_context(self, context: PythonModelContext):
        if os.name == 'nt':
            raise EvalFusionException('MLflow AI Gateway does not support Windows.')

        with open(context.artifacts[ARTIFACT_KEY_SETTINGS], 'r') as file:
            settings_dict = json.load(file)

        base_type = settings_dict['base_type']

        if isinstance(base_type, str):
            module_name, class_name = base_type.rsplit('.', 1)
            module = importlib.import_module(module_name)
            settings_dict['base_type'] = getattr(module, class_name)

        api_key = context.model_config.get('api_key')
        if api_key:
            settings_dict['kwargs']['api_key'] = api_key

        self._experiment_id = context.model_config.get('experiment_id')

        self.settings = EvalFusionLLMSettings(**settings_dict)
        self.__llm = self.settings.base_type(
            *self.settings.args, **self.settings.kwargs
        )

    def predict(self, context: PythonModelContext, model_input, params=None):
        prompt: str = (
            model_input.iloc[0, 0]
            if isinstance(model_input, DataFrame)
            else model_input
        )
        result = self.__llm.generate(prompt, use_json=False)
        token_usage = self.__llm.get_token_usage()

        with start_run(
            experiment_id=self._experiment_id, run_name='token-usage'
        ) as run:
            set_tag('purpose', 'token-usage')
            log_metric('input_tokens', token_usage.input)
            log_metric('output_tokens', token_usage.output)

        return [result]


set_model(MlFlowProxyLLM())
