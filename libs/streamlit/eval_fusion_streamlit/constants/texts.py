intro_text = """
    **Welcome to EvalFusion!**

    This app enables you to upload a JSON dataset for evaluating retrieval-augmented generation (RAG) models using RAGAS, DeepEval, MLflow, Phoenix Arize AI and TruLens under the hood. 
    
    Upload your dataset in the format specified below, and we'll evaluate it for you.

    #### Input Dataset Format:
    - The file should be a `.json` containing a list of records.
    - Each record should include the following fields:
        - `user_input`: The query or input from the user.
        - `retrieved_contexts`: A list of retrieved contexts.
        - `response`: The system's generated response.
        - `reference`: The ground truth or reference answer.

    Example:
    ```json
    [
        {
            "user_input": "What is the best framework for evaluating RAG applications?",
            "retrieved_contexts": [
                "Context 1...",
                "Context 2...",
                "Context 3..."
            ],
            "response": "EvalFusion, simply the best!",
            "reference": "The best evalution framework is EvalFusion."
        }
    ]
    ```
"""

footnote_text = """
    ---
    Check out our [GitHub repo](https://github.com/Luka958/eval-fusion/tree/main) and give it a ⭐️ if you like it. Thanks for your support!
"""
