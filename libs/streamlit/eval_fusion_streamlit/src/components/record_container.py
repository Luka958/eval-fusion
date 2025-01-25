import pandas as pd
import streamlit as st


def display_record_container(record: dict):
    record_text = f"""
        ### User input
        {repr(record["user_input"])}

        ### Retrieved contexts
        {repr(record["retrieved_contexts"])}

        ### Response
        {repr(record["response"])}

        ### Reference
        {repr(record["reference"])}
    """

    st.markdown(record_text)
