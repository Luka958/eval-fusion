from eval_fusion_test.utils.loader import load_evaluation_inputs
from eval_fusion_test.utils.nvidia import NvidiaLLM
from eval_fusion_trulens.evaluator import TruLensEvaluator
from eval_fusion_trulens.llm import TruLensProxyLLM


def test_evaluator():
    llm = NvidiaLLM('meta/llama-3.1-405b-instruct')
    proxy_llm = TruLensProxyLLM(llm)
    evaluator = TruLensEvaluator(proxy_llm)
    inputs = load_evaluation_inputs()

    inputs = inputs[:1]
    outputs = evaluator.evaluate(inputs, ...)

    for output in outputs:
        print(output)
