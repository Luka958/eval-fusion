from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    input: str
    output: str
    ground_truth: str
    relevant_chunks: list[str]


class EvaluationInputAlias(BaseModel):
    input: str
    output: str
    ground_truth: str
    relevant_chunks: str
