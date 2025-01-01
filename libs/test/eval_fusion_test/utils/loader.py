import json

from eval_fusion_core.models import EvaluationInput


def load_evaluation_inputs() -> list[EvaluationInput]:
    path = '../../assets/amnesty_qa.json'

    with open(path, 'r') as file:
        data = json.load(file)  # TODO

    return ...
