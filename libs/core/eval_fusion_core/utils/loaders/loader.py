import json

from pydantic import TypeAdapter

from eval_fusion_core.models import EvaluationInput


def load_evaluation_inputs(
    path: str, aliases: dict[str, str] | None = None
) -> list[EvaluationInput]:
    path = 'assets/amnesty_qa.json'

    with open(path, 'r') as file:
        raw_inputs: list[dict] = json.load(file)

    if aliases:
        raw_inputs = [
            dict((aliases.get(k, k), v) for k, v in x.items()) for x in raw_inputs
        ]

    return TypeAdapter(list[EvaluationInput]).validate_python(raw_inputs, strict=True)
