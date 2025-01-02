# TruLens

## Documentation
- [TruLens Documentation](https://www.trulens.org/getting_started/)

## Metrics

### Answer Relevance
Measures the relevance of the answer to the query, disregarding the context.

- Input
    - `query` - model's input
    - `response` - an answer
- Output
    - `score` - a float value
    
### Groundedness
Measures how well the answer relies on evidence from the context.

- Input
    - `response` - an answer
    - `contexts` - a given context
- Output
    - `score` - a float value

### Context Relevance
Measures the appropriateness and applicability of the answer with respect to the query and context.

- Input
    - `query` - model's input
    - `response` - an answer
    - `contexts` - a given context
- Output
    - `score` - a float value