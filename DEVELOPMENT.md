# Development

## Style guide

- use relative imports only for intra-package references, use absolute imports otherwise
- use single quotes

## Updating packages

- update package to a new version in let's say `eval-fusion-deepeval`, don't use `latest`: `poetry add deepeval@2.1.1` (update both `pyproject.toml` and `poetry.lock`, while `poetry update deepeval` updates only latter)
- then update the projects that rely on the updated project, let's say `eval-fusion-test`: `poetry update eval-fusion-deepeval`
