# pytest

## Basics
- test file: `poetry run pytest tests/deepeval/test_evaluator.py`
- test method: `poetry run pytest tests/deepeval/test_evaluator.py -k "test_evaluator"`

## Flags
- disable output capturing: `-s`
- disable warnings: `--disable-warnings`