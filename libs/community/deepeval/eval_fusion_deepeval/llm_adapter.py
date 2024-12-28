from deepeval.models import DeepEvalBaseLLM


class LLMAdapter(DeepEvalBaseLLM):
    def load_model(self):
        return ...  # TODO return model object

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        result = model.call(prompt)

        return ...

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return ...
