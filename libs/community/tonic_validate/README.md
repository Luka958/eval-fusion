# Tonic Validate

## Documentation
- [Tonic Validate Documentation](https://docs.tonic.ai/validate/) 

## Metrics

### Retrieval Precision
- Input  
    - `query`  
    - `contexts`

$$
\text{Retrieval Precision}
=
\frac{\sum_{i=1}^{M} \mathbf{1}\bigl[\text{context}_i\text{ relevant to query}\bigr]}{M}
$$

Whether the context retrieved is relevant to answer the given question. 

### Augmentation Accuracy
- Input  
    - `response`  
    - `contexts`

$$
\text{Augmentation Accuracy}
=
\frac{\sum_{i=1}^{M} \mathbf{1}\bigl[\text{context}_i\text{ appears in response}\bigr]}{M}
$$

Whether all of the retrieved context is included in the LLM answer.

### Augmentation Precision
- Input  
    - `query`  
    - `response`  
    - `contexts`

$$
R = \{\text{relevant contexts to query}\},\quad
U = \{\text{contexts in }R\text{ that appear in response}\}.

\text{Augmentation Precision}
=
\begin{cases}
\frac{|U|}{|R|}, & |R|>0,\\
0, & |R|=0.
\end{cases}
$$

Whether the relevant context is in the LLM answer.

### Answer Similarity
- Input  
    - `query`  
    - `reference`  
    - `response`

$$
\text{Answer Similarity}
=
\text{LLM‐assessed semantic similarity score on a 0–5 scale}
$$

How well the reference answer matches the LLM answer.

### Answer Consistency
- Input  
    - `response`  
    - `contexts`

$$
\{\text{main points}\} = \text{extract\_main\_points}(\text{response}),\quad
C_i = 
\begin{cases}
1, & \text{main\_point}_i\text{ supported by contexts},\\
0, & \text{otherwise}.
\end{cases}
$$

$$
\text{Answer Consistency}
=
\frac{\sum_{i=1}^{N} C_i}{N}
$$

Whether there is information in the LLM answer that does not come from the context.
