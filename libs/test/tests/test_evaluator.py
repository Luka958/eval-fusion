from eval_fusion_deepeval import DeepEvalEvaluator
from eval_fusion_test.utils.loader import load_evaluation_inputs
from eval_fusion_test.utils.vertexai.llm import VertexAILLM


def test_evaluator():
    llm = VertexAILLM('meta/llama-3.2-90b-vision-instruct-maas')
    evaluator = DeepEvalEvaluator(llm)
    inputs = load_evaluation_inputs()
    outputs = evaluator.evaluate(inputs, ...)

    for output in outputs:
        print(output)
