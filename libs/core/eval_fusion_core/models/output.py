from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvaluationOutputEntry(BaseModel):
    metric_name: str
    score: float
    reason: str | None


class EvaluationOutput(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    input_id: UUID
    output_entries: list[EvaluationOutputEntry]
