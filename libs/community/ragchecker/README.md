# RAGChecker

## Documentation
- [RAGChecker Documentation](https://github.com/amazon-science/RAGChecker)

## Metrics

### Claim Recall
- Input
    - `ground_truth`
    - `relevant_chunks`

$$
\text{Claim Recall}
=
\frac{|\{\text{claims in ground\_truth}\cap\text{relevant\_chunks}\}|}
     {|\{\text{claims in ground\_truth}\}|}
$$

### Context Precision
- Input
    - `ground_truth`
    - `relevant_chunks`

$$
\text{Context Precision}
=
\frac{|\{\text{relevant chunks}\}|}
     {|\{\text{all retrieved chunks}\}|}
$$

### Context Utilization
- Input
    - `output`
    - `ground_truth`
    - `relevant_chunks`

Let  
$$
A = \{\text{claims in ground\_truth entailed by any chunk}\},\quad
B = \{\text{claims in output}\cap A\}.
$$

$$
\text{Context Utilization}
=
\frac{|B|}
     {|A|}
$$

- $A$ – set of ground‑truth claims covered by any retrieved chunk  
- $B$ – set of claims in the model output drawn from $A$

### Precision
- Input
    - `output`
    - `ground_truth`

$$
\text{Precision}
=
\frac{|\{\text{correct claims in output}\}|}
     {|\{\text{all claims in output}\}|}
$$

### Recall
- Input
    - `output`
    - `ground_truth`

$$
\text{Recall}
=
\frac{|\{\text{correct claims in output}\}|}
     {|\{\text{all claims in ground\_truth}\}|}
$$

### F1
- Input
    - `output`
    - `ground_truth`

$$
\text{F1}
=
2 \times \frac{\text{Precision}\times\text{Recall}}
              {\text{Precision}+\text{Recall}}
$$

### Faithfulness
- Input
    - `output`
    - `relevant_chunks`

$$
\text{Faithfulness}
=
\frac{|\{\text{claims in output entailed by any chunk}\}|}
     {|\{\text{all claims in output}\}|}
$$

### Hallucination
- Input
    - `output`
    - `relevant_chunks`

$$
\text{Hallucination}
=
\frac{|\{\text{incorrect claims in output unsupported by any chunk}\}|}
     {|\{\text{all claims in output}\}|}
$$

### Noise Sensitivity (Relevant)
- Input
    - `output`
    - `relevant_chunks`

$$
\text{Noise Sensitivity (Relevant)}
=
\frac{|\{\text{incorrect claims in output coming from relevant chunks}\}|}
     {|\{\text{all claims in output}\}|}
$$

### Noise Sensitivity (Irrelevant)
- Input
    - `output`
    - `relevant_chunks`

$$
\text{Noise Sensitivity (Irrelevant)}
=
\frac{|\{\text{incorrect claims in output coming from irrelevant chunks}\}|}
     {|\{\text{all claims in output}\}|}
$$

### Self Knowledge
- Input
    - `output`
    - `ground_truth`
    - `relevant_chunks`

$$
\text{Self Knowledge}
=
\frac{|\{\text{correct claims in output not found in any retrieved chunk}\}|}
     {|\{\text{all claims in output}\}|}
$$
