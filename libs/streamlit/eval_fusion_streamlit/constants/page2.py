options = {
    '***All available metrics***': 0,
    '***Semantic Groups***': 1,
    '***Frameworks & Metrics***': 2,
}

framework_metrics = {
    'RAGAS': [
        'Context Precision',
        'Context Recall',
        'Context Entity Recall',
        'Noise Sensitivity',
        'Response Relevancy',
        'Faithfulness',
    ],
    'DeepEval': [
        'Answer Relevancy',
        'Faithfulness',
        'Contextual Precision',
        'Contextual Recall',
        'Contextual Relevancy',
    ],
    'MLflow': [
        'Answer Correctness',
        'Answer Relevance',
        'Answer Similarity',
        'Faithfulness',
        'Relevance',
    ],
    'Phoenix Arize AI': [
        'Retrieval (RAG) Relevance',
        'Hallucination',
        'Q&A on Retrieved Data',
    ],
    'TruLens': ['Answer Relevance', 'Groundedness', 'Context Relevance'],
}

semantic_groups = {
    'Group 1': {
        'RAGAS': ['Response Relevancy'],
        'DeepEval': ['Answer Relevancy'],
        'MLflow': ['Answer Relevance'],
        'TruLens': ['Answer Relevance'],
    },
    'Group 2': {
        'RAGAS': ['Faithfulness'],
        'DeepEval': ['Faithfulness'],
        'MLflow': ['Faithfulness'],
        'TruLens': ['Groundedness'],
    },
    'Group 3': {'RAGAS': ['Context Precision'], 'DeepEval': ['Contextual Precision']},
    'Group 4': {'RAGAS': ['Context Recall'], 'DeepEval': ['Contextual Recall']},
    'Group 5': {
        'DeepEval': ['Contextual Relevancy'],
        'Phoenix Arize AI': ['Retrieval (RAG) Relevance'],
        'TruLens': ['Context Relevance'],
    },
    'Group 6': {'MLflow': ['Relevance'], 'Phoenix Arize AI': ['Hallucination']},
}


choose_metrics_text = """
    ## Step 2️⃣: &nbsp;**Choose frameworks and metrics**.
    - You can choose metrics for evaluation based on their semantic similarity and meaning by selecting ***Semantic Groups***
    or organize them by evaluation frameworks by selecting ***Frameworks & Metrics***. 
    If you prefer, you can also select ***All available metrics*** for a comprehensive evaluation.
"""

semantic_groups_text = """
    #### Semantic Groups
    - Here is an overview of semantic groups with the metrics they offer:

    |***Semantic groups***  | **RAGAS**               | **DeepEval**           | **MLflow**            | **Phoenix Arize AI**        | **TruLens**        |
    |:---------------------:|:-----------------------:|:----------------------:|:---------------------:|:---------------------------:|:------------------:|
    |        (1)            | Response Relevancy      | Answer Relevancy       | Answer Relevance      |                             | Answer Relevance   |
    |        (2)            | Faithfulness            | Faithfulness           | Faithfulness          |                             | Groundedness       |
    |        (3)            | Context Precision       | Contextual Precision   |                       |                             |                    |
    |        (4)            | Context Recall          | Contextual Recall      |                       |                             |                    |
    |        (5)            |                         | Contextual Relevancy   |                       | Retrieval  Relevance        | Context Relevance  |
    |        (6)            |                         |                        | Relevance             | Hallucination               |                    |

"""

frameworks_metrics_text = """
    #### Frameworks & Metrics
    - First, select the frameworks you want to use for evaluation, and then choose the metrics for each of them.
"""

next_button = """
    Click "Next" to view the results.
"""
