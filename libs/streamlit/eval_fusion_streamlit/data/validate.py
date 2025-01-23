from typing import List

from pydantic import BaseModel, ValidationError


class Record(BaseModel):
    user_input: str
    retrieved_contexts: List[str]
    response: str
    reference: str


class DatasetSchema(BaseModel):
    records: List[Record]


def validate_dataset_format(data: list) -> bool:
    try:
        DatasetSchema(records=data)
    except ValidationError:
        return False

    return True
