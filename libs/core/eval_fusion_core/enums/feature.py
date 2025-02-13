from enum import Enum


class Feature(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    GROUND_TRUTH = 'ground_truth'
    RELEVANT_CHUNKS = 'relevant_chunks'
    ALL = 'all'
