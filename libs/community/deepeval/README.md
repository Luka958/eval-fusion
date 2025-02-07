# DeepEval

## Documentation
- [DeepEval Documentation](https://docs.confident-ai.com/docs/metrics-introduction)

## Metrics

### Answer Relevancy
- Input
    - `input`
    - `actual_output`

$$
\text{Answer Relevancy} = \frac{\text{Number of Relevant Statements}}{\text{Total Number of Statements}}
$$

### Faithfulness
- Input
    - `input`
    - `actual_output`
    - `retrieval_context`

$$
\text{Faithfulness} = \frac{\text{Number of Truthful Claims}}{\text{Total Number of Claims}}
$$

### Contextual Precision
- Input
    - `input`
    - `actual_output`
    - `expected_output`
    - `retrieval_context`

$$
\text{Contextual Precision} = \frac{1}{\text{Number of Relevant Nodes}} \sum_{k=1}^{n} \left( \frac{\text{Number of Relevant Nodes Up to Position } k}{k} \times r_k \right)
$$

- $k$ is the $(i+1)^{\text{th}}$ node in the `retrieval_context`
- $n$ is the length of the `retrieval_context`
- $r_k$ is the binary relevance for the kth node in the `retrieval_context`, $r_k = 1$ for nodes that are relevant, $0$ if not

### Contextual Recall
- Input
    - `input`
    - `actual_output`
    - `expected_output`
    - `retrieval_context`

$$
\text{Contextual Recall} = \frac{\text{Number of Attributable Statements}}{\text{Total Number of Statements}}
$$

### Contextual Relevancy
- Input
    - `input`
    - `actual_output`
    - `retrieval_context`

$$
\text{Contextual Relevancy} = \frac{\text{Number of Relevant Statements}}{\text{Total Number of Statements}}
$$