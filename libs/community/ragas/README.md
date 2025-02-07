# Ragas

## Documentation
- [Ragas Documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

## Metrics

### Context Precision
- Input
    - `user_input`
    - `retrieved_contexts`
    - `reference`

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@}k \times v_k \right)}{\text{Total number of relevant items in the top K results}}
$$

$$
\text{Precision@}k = \frac{\text{true positives@}k}{\left(\text{true positives@}k + \text{false positives@}k\right)}
$$

- $K$ - the total number of chunks in `retrieved_contexts`
- $v_k$ - relevance indicator at rank $k$

### Context Recall
- Input
    - `user_input`
    - `retrieved_contexts`
    - `reference`

$$
\text{context recall} = \frac{|\text{GT claims that can be attributed to context}|}{|\text{Number of claims in GT}|}
$$

### Context Entity Recall
- Input
    - `retrieved_contexts`
    - `reference`

$$
\text{context entity recall} = \frac{|CE \cap GE|}{|GE|}
$$

- $GE$ - set of entities present in `reference`
- $CE$ - set of entities present in `retrieved_contexts`

### Noise Sensitivity
- Input
    - `user_input`
    - `retrieved_contexts`
    - `response`
    - `reference`

$$
\text{noise sensitivity (relevant)} = \frac{|\text{Total number of incorrect claims in response}|}{|\text{Total number of claims in the response}|}
$$

### Response Relevancy
- Input
    - `user_input`
    - `response`

$$
\text{response relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{g_i}, E_o)
$$

- $E_{g_i}$ - the embedding of the generated question 
- $E_o$ - the embedding of the original question
- $N$ - the number of generated questions, which is 3 default

### Faithfulness
- Input
    - `retrieved_contexts`
    - `response`

$$
\text{Faithfulness} = \frac{|\text{Number of claims in the generated answer that can be inferred from given context}|}{|\text{Total number of claims in the generated answer}|}
$$
