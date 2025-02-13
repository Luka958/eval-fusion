from enum import Enum


class EvaluationInputFeature(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    GROUND_TRUTH = 'ground_truth'
    RELEVANT_CHUNKS = 'relevant_chunks'
    ALL = 'all'
