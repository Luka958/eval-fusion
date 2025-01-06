from eval_fusion_deepeval.evaluator import DeepEvalEvaluator
from eval_fusion_deepeval.llm import DeepEvalLLM
from eval_fusion_test.utils.loader import load_evaluation_inputs
from eval_fusion_test.utils.nvidia import NvidiaLLM


def test_evaluator():
    # llm_delegate = VertexAILLM('meta/llama-3.2-90b-vision-instruct-maas')
    llm_delegate = NvidiaLLM('meta/llama-3.1-405b-instruct')
    llm = DeepEvalLLM(llm_delegate)
    evaluator = DeepEvalEvaluator(llm)
    inputs = load_evaluation_inputs()

    inputs = inputs[:1]
    outputs = evaluator.evaluate(inputs, ...)

    for output in outputs:
        print(output)
