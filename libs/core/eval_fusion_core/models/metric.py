from pydantic import BaseModel


class InputUsage:
    input: int


class InputOutputUsage:
    input: int
    output: int


class MetricUsage(BaseModel):
    llm: InputOutputUsage
    embedding: InputUsage


class Metric(BaseModel):
    name: str
    score: float  # is it Metric or EvaluationResult property?
    usage: MetricUsage  # is it Metric or EvaluationResult property?


# TODO separate metrics by types -> LLM, embedding, etc.
