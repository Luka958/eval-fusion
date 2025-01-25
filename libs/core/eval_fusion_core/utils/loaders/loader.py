import json

from pydantic import TypeAdapter

from eval_fusion_core.models import EvaluationInput, EvaluationInputAlias


def load_evaluation_inputs(
    path: str, alias: EvaluationInputAlias = None
) -> list[EvaluationInput]:
    with open(path, 'r') as file:
        raw_inputs: list[dict] = json.load(file)

    if alias:
        alias_dict = {v: k for k, v in alias.model_dump().items()}
        raw_inputs = [
            {alias_dict.get(k, k): v for k, v in x.items()} for x in raw_inputs
        ]

    return TypeAdapter(list[EvaluationInput]).validate_python(raw_inputs, strict=True)
