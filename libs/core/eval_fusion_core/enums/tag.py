from enum import Enum


class MetricTag(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    GROUND_TRUTH = 'ground_truth'
    RELEVANT_CHUNKS = 'relevant_chunks'
    ALL = 'all'
