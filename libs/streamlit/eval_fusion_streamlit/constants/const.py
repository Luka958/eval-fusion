openai_llm_models = ['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini']
openai_embedding_models = ['text-embedding-3-small', 'text-embedding-3-large']

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
