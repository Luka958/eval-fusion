from eval_fusion_deepeval import DeepEvalEvaluator

from .vertex_ai_llm import VertexAILLM


def load_evaluation_inputs():
    import json

    path = '../assets/amnesty_qa.json'

    with open(path, 'r') as file:
        data = json.load(file)


def test_evaluator():
    llm = VertexAILLM('meta/llama-3.2-90b-vision-instruct-maas')
    evaluator = DeepEvalEvaluator(llm)
    inputs = ...
    outputs = evaluator.evaluate(inputs, ...)

    for output in outputs:
        print(output)
