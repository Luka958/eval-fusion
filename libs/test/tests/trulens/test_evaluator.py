from decouple import config
from eval_fusion_core.utils.loaders import load_evaluation_inputs
from eval_fusion_openai import OpenAILLM
from eval_fusion_trulens.evaluator import TruLensEvaluator
from eval_fusion_trulens.llm import TruLensProxyLLM


NVIDIA_BASE_URL = config('NVIDIA_BASE_URL')
NVIDIA_API_KEY = config('NVIDIA_API_KEY')


def test_evaluator():
    llm = OpenAILLM('meta/llama-3.1-405b-instruct', NVIDIA_BASE_URL, NVIDIA_API_KEY)
    proxy_llm = TruLensProxyLLM(llm)
    evaluator = TruLensEvaluator(proxy_llm)
    inputs = load_evaluation_inputs('assets/amnesty_qa.json')

    inputs = inputs[:1]
    outputs = evaluator.evaluate(inputs, ...)

    for output in outputs:
        print(output)
