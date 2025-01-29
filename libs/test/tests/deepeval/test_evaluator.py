from decouple import config
from eval_fusion_core.enums import MetricTag
from eval_fusion_core.models.settings import EvalFusionLLMSettings
from eval_fusion_core.utils.loaders import load_evaluation_inputs
from eval_fusion_deepeval.evaluator import DeepEvalEvaluator
from eval_fusion_openai import OpenAILLM


NVIDIA_BASE_URL = config('NVIDIA_BASE_URL')
NVIDIA_API_KEY = config('NVIDIA_API_KEY')


def test_evaluator():
    settings = EvalFusionLLMSettings(
        base_type=OpenAILLM,
        kwargs={
            'model_name': 'meta/llama-3.1-405b-instruct',
            'base_url': NVIDIA_BASE_URL,
            'api_key': NVIDIA_API_KEY,
        },
    )
    evaluator = DeepEvalEvaluator(settings)
    inputs = load_evaluation_inputs('assets/amnesty_qa.json')

    inputs = inputs[:1]
    outputs = evaluator.evaluate_by_tag(inputs, MetricTag.ALL)

    for output in outputs:
        print(output)
