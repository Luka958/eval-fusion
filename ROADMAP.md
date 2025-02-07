# Roadmap

- add ARES
- add llama-index
- add RAGChecker
- add arxiv links if available for each framework
- concurrency
- rework metrics
  - use str Enum for metrics
  - map Enum values to functions/classes
- renaming
  - MetricTag -> Feature (each EvaluationInput consists of features: input, output, etc.)
  - EvaluationInput -> Sample (each Sample consists of Features - common ML convention)
  - EvaluationOutput and EvaluationOutputEntry -> TODO
