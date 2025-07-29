# eval-fusion

## Setup

### Install dependencies only
`python3 setup.py`

### Install dependencies and create VSCode workspace
`python3 setup.py vscode`

## Format manually
`ruff check . --fix`

### Install
`poetry run pre-commit install`

Run this after each change to `.pre-commit-config.yaml`.

### Verify
`poetry run pre-commit run --all-files`

## Requirements
| Package         | Version   | Requirements                                          |
|-----------------|-----------|-------------------------------------------------------|
| deepeval        | 2.0.9     | <3.13, >=3.9                                          |
| ragas           | 0.2.9     | >=3.9                                                 |
| arize-phoenix   | 7.3.2     | <3.14, >=3.9                                          |
| mlflow          | 2.19.0    | >=3.9                                                 |
| trulens         | 1.2.11    | >=3.8                                                 |
| ragchecker      | 0.1.9     | <4.0, >=3.9                                           |
| llama-index     | 0.12.17   | <4.0, >=3.9                                           |
| tonic-validate  | 2.0.0     | >=3.8.1                                               |
| streamlit       | 1.41.1    | >=3.9, <3.9.7 \|\| >3.9.7, <3.13                      |

## Overview
|                         | LLM | Embedding | Input | Relevant Chunks | Output | Ground Truth |
|-------------------------|-----|-----------|-------|------------------|--------|---------------|
| **Ragas**               |     |           |       |                  |        |               |
| Context Precision       | Yes | No        | Yes   | Yes              | No     | Yes           |
| Context Recall          | Yes | No        | Yes   | Yes              | No     | Yes           |
| Context Entity Recall   | Yes | No        | No    | Yes              | No     | Yes           |
| Noise Sensitivity       | Yes | No        | Yes   | Yes              | Yes    | Yes           |
| Response Relevancy      | Yes | Yes       | Yes   | No               | Yes    | No            |
| Faithfulness            | Yes | No        | No    | Yes              | Yes    | No            |
| **DeepEval**            |     |           |       |                  |        |               |
| Answer Relevancy        | Yes | No        | Yes   | No               | Yes    | No            |
| Faithfulness            | Yes | No        | Yes   | Yes              | Yes    | No            |
| Contextual Precision    | Yes | No        | Yes   | Yes              | Yes    | Yes           |
| Contextual Recall       | Yes | No        | Yes   | Yes              | Yes    | Yes           |
| Contextual Relevancy    | Yes | No        | Yes   | Yes              | Yes    | No            |
| **MLflow**              |     |           |       |                  |        |               |
| Answer Correctness      | Yes | No        | Yes   | No               | Yes    | Yes           |
| Answer Relevance        | Yes | No        | Yes   | No               | Yes    | Yes           |
| Answer Similarity       | Yes | No        | Yes   | No               | Yes    | Yes           |
| Faithfulness            | Yes | No        | Yes   | Yes              | Yes    | No            |
| Relevance               | Yes | No        | Yes   | Yes              | Yes    | No            |
| **Phoenix Arize AI**    |     |           |       |                  |        |               |
| Relevance               | Yes | No        | Yes   | Yes              | No     | No            |
| Hallucination           | Yes | No        | Yes   | Yes              | Yes    | No            |
| Q&A                     | Yes | No        | Yes   | Yes              | Yes    | No            |
| **TruLens**             |     |           |       |                  |        |               |
| Answer Relevance        | Yes | No        | Yes   | No               | Yes    | No            |
| Groundedness            | Yes | No        | No    | Yes              | Yes    | No            |
| Context Relevance       | Yes | No        | Yes   | Yes              | Yes    | No            |
| **RAGChecker**          |     |           |       |                  |        |               |
| Claim Recall            | No  | No        | No    | Yes              | No     | Yes           |
| Context Precision       | No  | No        | No    | Yes              | No     | Yes           |
| Context Utilization     | No  | No        | No    | Yes              | Yes    | Yes           |
| Faithfulness            | No  | No        | No    | Yes              | Yes    | No            |
| Hallucination           | No  | No        | No    | Yes              | Yes    | No            |
| Noise Sensitivity In Relevant | No  | No        | No    | Yes              | Yes    | No            |
| Noise Sensitivity In Irrelevant| No | No        | No    | Yes              | Yes    | No            |
| Self Knowledge          | No  | No        | No    | Yes              | Yes    | Yes           |
| **LlamaIndex**          |     |           |       |                  |        |               |
| Answer Relevancy        | No  | No        | Yes   | No               | Yes    | No            |
| Context Relevancy       | No  | No        | Yes   | Yes              | No     | No            |
| Correctness             | No  | No        | Yes   | No               | Yes    | Yes           |
| Relevancy               | No  | No        | Yes   | Yes              | Yes    | No            |
| Faithfulness            | No  | No        | No    | Yes              | Yes    | No            |
| Semantic Similarity     | No  | No        | No    | No               | Yes    | Yes           |
| **Tonic Validate**      |     |           |       |                  |        |               |
| Augmentation Precision  | No  | No        | Yes   | Yes              | Yes    | No            |
| Answer Similarity       | No  | No        | Yes   | No               | Yes    | Yes           |
| Retrieval Precision     | No  | No        | Yes   | Yes              | No     | No            |
| Answer Consistency      | No  | No        | No    | Yes              | Yes    | No            |
| Augmentation Accuracy   | No  | No        | No    | Yes              | Yes    | No            |
