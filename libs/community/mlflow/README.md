# MLflow

## Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Metrics

### Answer Correctness
Measures the factual correctness of the model’s output relative to the ground truth.

- Input
    - `inputs` - queries
    - `outputs` - model's outputs
    - `predictions` - ground truths
- Output
    - `score` - a float value
    - `justification` - a textual explanation of the score

### Answer Relevance
Measures the relevance of the model’s output to the input, disregarding the context.

- Input
    - `inputs` - queries
    - `outputs` - model's outputs
- Output
    - `score` - a float value
    - `justification` - a textual explanation of the score

### Answer Similarity
Measures how similar the model’s output is to the ground truth.

- Input
    - `inputs` - queries
    - `outputs` - model's outputs
    - `predictions` - ground truths
- Output
    - `score` - a float value
    - `justification` - a textual explanation of the score
    
### Faithfulness
Measures the faithfulness of the model’s output concerning the provided context.

- Input
    - `inputs` - queries
    - `outputs` - model's outputs
    - `context` - a given context
- Output
    - `score` - a float value
    - `justification` - a textual explanation of the score

### Relevance
Measures the appropriateness and applicability of the model's output with respect to the input and context.

- Input
    - `inputs` - queries
    - `outputs` - model's outputs
    - `context` - a given context
- Output
    - `score` - a float value
    - `justification` - a textual explanation of the score