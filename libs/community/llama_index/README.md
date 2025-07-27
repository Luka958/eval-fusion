# LlamaIndex

## Documentation
- [LlamaIndex Evaluation API](https://docs.llamaindex.ai/en/stable/api_reference/evaluation/)

## Metrics

### Answer Relevancy
- Input
    - `query`
    - `response`

Evaluates how well the generated response addresses the input query by comparing response content to query intent using a judge LLM.

### Context Relevancy
- Input
    - `query`
    - `contexts`

Assesses whether each retrieved context passage contains information pertinent to the input query.

### Correctness
- Input
    - `query`
    - `response`
    - `reference`

Measures the factual accuracy of the generated response against a ground‐truth reference answer by scoring its correctness.

### Faithfulness
- Input
    - `response`
    - `contexts`

Determines whether all claims in the response are supported by the provided contexts, identifying any hallucinated content.

### Relevancy
- Input
    - `query`
    - `response`
    - `contexts`

Evaluates the combined relevancy of both the retrieved contexts and the generated response to the input query, ensuring the response is grounded in the contexts and answers the query.

### Semantic Similarity
- Input
    - `response`
    - `reference`

Computes the semantic alignment between the generated response and the reference answer by comparing their embeddings.
