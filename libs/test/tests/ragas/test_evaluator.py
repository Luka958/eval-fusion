from eval_fusion_ragas.evaluator import RagasEvaluator
from eval_fusion_ragas.llm import RagasLLM
from eval_fusion_test.utils.loader import load_evaluation_inputs
from eval_fusion_test.utils.nvidia import NvidiaLLM


def test_evaluator():
    llm_delegate = NvidiaLLM('meta/llama-3.1-405b-instruct')
    llm = RagasLLM(llm_delegate)
    evaluator = RagasEvaluator(llm)
    inputs = load_evaluation_inputs()

    inputs = inputs[:1]
    outputs = evaluator.evaluate(inputs, ...)

    for output in outputs:
        print(output)
