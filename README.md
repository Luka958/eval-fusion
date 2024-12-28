# eval-fusion

## setup
`.\setup.ps1`

## pre-commit

### install
`poetry run pre-commit install`

Run this after each change to `.pre-commit-config.yaml`.

### verify
`poetry run pre-commit run --all-files`

## requirements
| Package         | Version   | Requirements                                          |
|-----------------|-----------|-------------------------------------------------------|
| deepeval        | 2.0.9     | <3.13, >=3.9                                          |
| ragas           | 0.2.9     | >=3.9                                                 |
| arize-phoenix   | 7.3.2     | <3.14, >=3.9                                          |
| mlflow          | 2.19.0    | >=3.9                                                 |
| trulens         | 1.2.11    | >=3.8 |
