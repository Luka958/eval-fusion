intro_text = """
    # Welcome to **Eval Fusion!**

    This app allows you to easily evaluate Retrieval-Augmented Generation (RAG) models using 
    RAGAS, DeepEval, MLflow, Phoenix Arize AI and TruLens under the hood. Let's start!
    
    ## 🚀 Get Started
    ### -> Step 1️⃣: &nbsp;**Upload your dataset** in JSON format.
    #### Input Dataset Format:
    - The file should be a `.json` containing a list of records.
    - Each record should include the following fields:
        - `input`: The query or input from the user.
        - `relevant_chunks`: A list of retrieved contexts.
        - `output`: The system's generated response.
        - `ground_truth`: The ground truth or reference answer.

    Example:
    ```json
    [
        {
            "input": "What is the best library for evaluating RAG applications?",
            "relevant_chunks": [
                "Context 1...",
                "Context 2...",
                "Context 3..."
            ],
            "output": "EvalFusion, simply the best!",
            "ground_truth": "The best evaluation library is EvalFusion."
        }
    ]
    ```

    ### -> Step 2️⃣: &nbsp;**Choose frameworks and metrics**.
    - Currently, there are 5 evaluation frameworks available, and here is an overview of them along with the metrics they offer:

    | **RAGAS**                        | **DeepEval**                | **MLflow**                  | **Phoenix Arize AI**        | **TruLens**               |
    |:--------------------------------:|:---------------------------:|:---------------------------:|:---------------------------:|:-------------------------:|
    | Context Precision                | Answer Relevancy            | Answer Correctness          | Retrieval (RAG) Relevance   | Answer Relevance          |
    | Context Recall                   | Faithfulness                | Answer Relevance            | Hallucination               | Groundedness              |
    | Context Entity Recall            | Contextual Precision        | Answer Similarity           | Q&A on Retrieved Data       | Context Relevance         |
    | Noise Sensitivity                | Contextual Recall           | Faithfulness                |                             |                           |
    | Response Relevancy               | Contextual Relevancy        | Relevance                   |                             |                           |
    | Faithfulness                     |                             |                             |                             |                           |

    ### -> Step 3️⃣: &nbsp;**View the evaluation results**.
    - Finally, you can now review the results returned by the metrics, the number of tokens used, and the execution time.
    
    <br/>
"""

next_button = """
    Ready? Click "Next" to continue!
"""

footnote_text = """
    ---
    Check out our [GitHub repo](https://github.com/Luka958/eval-fusion/tree/main) and give it a ⭐️ if you like it. Thanks for your support!
"""
