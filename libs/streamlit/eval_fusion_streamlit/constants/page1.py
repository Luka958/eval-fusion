openai_llm_models = ['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini']
openai_embedding_models = ['text-embedding-3-small', 'text-embedding-3-large']

prerequisites_text = """
   ## ⚙️ &nbsp;**Prerequisites** 
   - Before we begin, you’ll need to provide your OpenAI API key and select models to proceed with the evaluation.
"""

upload_dataset_text = """
    ## Step 1️⃣: &nbsp;**Upload your dataset** in JSON format
    - Each record in your dataset will be assigned a unique ID corresponding to its position 
    in the dataset, starting from 1. This way, after evaluation, you’ll be able to easily reference 
    and locate specific examples in the ***Inspect records*** section.
"""

preview_text = """
    ### Preview of the records
"""

inspect_text = """
    ### Inspect records
"""

next_button = """
    Click "Next" to choose metrics.
"""
