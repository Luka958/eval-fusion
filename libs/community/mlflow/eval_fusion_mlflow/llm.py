from eval_fusion_core.models.settings import EvalFusionLLMSettings
from mlflow.pyfunc.model import PythonModel, PythonModelContext


class MlFlowProxyLLM(PythonModel):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.settings = settings

    def load_context(self, context: PythonModelContext):
        self.__llm = self.settings.base_type(
            *self.settings.args, **self.settings.kwargs
        )

    def predict(self, context: PythonModelContext, model_input: list[str]) -> list[str]:
        assert len(model_input) == 0

        result = self.__llm.generate(model_input[0])

        return [result]
