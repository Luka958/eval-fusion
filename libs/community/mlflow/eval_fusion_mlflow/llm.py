import os

from eval_fusion_core.exceptions import EvalFusionException
from eval_fusion_core.models import TokenUsage
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from pandas import DataFrame


class MlFlowProxyLLM(PythonModel):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.settings = settings

        if os.name == 'nt':
            raise EvalFusionException('MLflow AI Gateway does not support Windows.')

    def load_context(self, context: PythonModelContext):
        self.__llm = self.settings.base_type(
            *self.settings.args, **self.settings.kwargs
        )

    def predict(self, context, model_input, params=None):
        # NOTE: type hints are not allowed here!
        if params:
            temperature: float = params['temperature']
            n: int = params['n']
            max_tokens: int = params['max_tokens']
            top_p: float = params['top_p']

        prompt: str = (
            model_input.iloc[0, 0]
            if isinstance(model_input, DataFrame)
            else model_input
        )
        result = self.__llm.generate(prompt, use_json=False)

        return [result]

    def get_token_usage(self) -> TokenUsage:
        return self.__llm.get_token_usage()
