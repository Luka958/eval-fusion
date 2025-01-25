# eval-fusion

## Setup
`.\setup.ps1`

## Format manually
`ruff check . --fix`

### Install
`poetry run pre-commit install`

Run this after each change to `.pre-commit-config.yaml`.

### Verify
`poetry run pre-commit run --all-files`

## Requirements
| Package         | Version   | Requirements                                          |
|-----------------|-----------|-------------------------------------------------------|
| deepeval        | 2.0.9     | <3.13, >=3.9                                          |
| ragas           | 0.2.9     | >=3.9                                                 |
| arize-phoenix   | 7.3.2     | <3.14, >=3.9                                          |
| mlflow          | 2.19.0    | >=3.9                                                 |
| trulens         | 1.2.11    | >=3.8                                                 |
| streamlit       | 1.41.1    | >=3.9, <3.9.7 \|\| >3.9.7, <3.13                      |