from decouple import config

from eval_fusion_test.utils.openai import OpenAILLM


NVIDIA_BASE_URL = config('NVIDIA_BASE_URL')
NVIDIA_API_KEY = config('NVIDIA_API_KEY')


class NvidiaLLM(OpenAILLM):
    def __init__(self, model_name: str):
        super().__init__(model_name, NVIDIA_BASE_URL, NVIDIA_API_KEY)
