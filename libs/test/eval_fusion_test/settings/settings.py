from decouple import config
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from eval_fusion_openai import OpenAILLM


def get_openai_settings():
    return EvalFusionLLMSettings(
        base_type=OpenAILLM,
        kwargs={
            'model_name': 'gpt-4o-mini',
            'base_url': config('OPENAI_BASE_URL'),
            'api_key': config('OPENAI_API_KEY'),
        },
    )
