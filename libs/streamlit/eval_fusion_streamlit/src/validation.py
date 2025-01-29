from typing import List

from pydantic import BaseModel, ValidationError


class Record(BaseModel):
    input: str
    relevant_chunks: List[str]
    output: str
    ground_truth: str


class DatasetSchema(BaseModel):
    records: List[Record]


def validate_dataset_format(data: list) -> bool:
    try:
        DatasetSchema(records=data)
    except ValidationError:
        return False

    return len(data) >= 5
