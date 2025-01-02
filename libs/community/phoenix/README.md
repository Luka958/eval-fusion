# Phoenix

## Documentation
- [Arize AI Phoenix Documentation](https://docs.arize.com/phoenix/)

## Metrics

### Retrieval (RAG) Relevance
Evaluates whether a retrieved document is _relevant or irrelevant_ to the corresponding query.

- Input
    - `input` - a query
    - `reference` - a retrieved document
- Output
    - `label` - a binary value ($relevant | unrelated$)

### Hallucination
Evaluates whether a response is _a hallucination_ given a query and one or more retrieved documents.

- Input
    - `input` - a query
    - `output` - a response
    - `reference` - one or more retrieved documents
- Output
    - `label` - a binary value ($hallucinated | factual$)

### Q&A on Retrieved Data
Evaluates whether a response is _correct or incorrect_ given a query and one or more retrieved documents.

- Input
    - `input` - a query
    - `output` - a response
    - `reference` - one or more retrieved documents
- Output
    - `label` - a binary value ($correct | incorrect$)