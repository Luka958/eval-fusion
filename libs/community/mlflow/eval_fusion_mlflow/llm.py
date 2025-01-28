from eval_fusion_core.models import EvalFusionLLMSettings
from mlflow.pyfunc.model import PythonModel, PythonModelContext


class MlFlowProxyLLM(PythonModel):
    def __init__(self, settings: EvalFusionLLMSettings):
        self.settings = settings

    def load_context(self, context: PythonModelContext):
        self.llm_delegate = self.settings.base_type(
            *self.settings.args, **self.settings.kwargs
        )

    def predict(self, context: PythonModelContext, model_input: list[str]) -> list[str]:
        completion = self.client.chat.completions.create(
            model='gpt-4o-mini', messages=[{'role': 'user', 'content': model_input[0]}]
        )
        answer = completion.choices[0].message.content

        return [answer]
