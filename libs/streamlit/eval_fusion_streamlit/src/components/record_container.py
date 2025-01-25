import pandas as pd
import streamlit as st


def display_record_container(record: dict):
    record_text = f"""
        ### Input
        {repr(record["input"])}

        ### Relevant chunks
        {repr(record["relevant_chunks"])}

        ### Output
        {repr(record["output"])}

        ### Ground truth
        {repr(record["ground_truth"])}
    """

    st.markdown(record_text)
