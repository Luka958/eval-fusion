# Development

## Style guide

- use relative imports only for intra-package references, use absolute imports otherwise
- use single quotes

## Updating packages

- update package to a new version in let's say `eval-fusion-deepeval`, don't use `latest`: `uv add deepeval@2.1.1` (update both `pyproject.toml` and `uv.lock`, while `uv update deepeval` updates only latter)
- then update the projects that rely on the updated project, let's say `eval-fusion-test`: `uv sync --upgrade-package eval-fusion-deepeval
`
