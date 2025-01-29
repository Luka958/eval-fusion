from decouple import config
from eval_fusion_core.models.settings import EvalFusionEMSettings, EvalFusionLLMSettings
from eval_fusion_openai import OpenAIEM, OpenAILLM


def get_openai_settings():
    llm_settings = EvalFusionLLMSettings(
        base_type=OpenAILLM,
        kwargs={
            'model_name': 'gpt-4o-mini',
            'base_url': config('OPENAI_BASE_URL'),
            'api_key': config('OPENAI_API_KEY'),
        },
    )
    em_settings = EvalFusionEMSettings(
        base_type=OpenAIEM,
        kwargs={
            'model_name': 'text-embedding-3-small',
            'base_url': config('OPENAI_BASE_URL'),
            'api_key': config('OPENAI_API_KEY'),
        },
    )

    return llm_settings, em_settings
