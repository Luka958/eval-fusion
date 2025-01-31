from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class EvaluationOutputEntry(BaseModel):
    metric_name: str
    score: float | None
    reason: str | None
    error: str | None
    time: float | None

    @model_validator(mode='after')
    def check(cls, model: 'EvaluationOutputEntry') -> 'EvaluationOutputEntry':
        if model.score is not None:
            if model.error is not None:
                raise ValueError('If "score" exists, "error" must not exist.')

        else:
            if model.reason is not None:
                raise ValueError('If "score" does not exist, "reason" must not exist.')

            if model.error is None:
                raise ValueError('If "score" does not exist, "error" must exist.')

        if model.reason is not None:
            if model.score is None:
                raise ValueError('If "reason" exists, "score" must also exist.')

            if model.error is not None:
                raise ValueError('If "reason" exists, "error" must not exist.')

        return model


class EvaluationOutput(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    input_id: UUID
    output_entries: list[EvaluationOutputEntry]
