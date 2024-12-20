from pydantic import BaseModel


class EvaluationInput(BaseModel):
    input: str
    output: str
    ground_truth: str
    relevant_chunks: list[str]


class EvaluationOutput(EvaluationInput):
    pass
