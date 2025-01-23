import json

import pandas as pd
import streamlit as st

from constants.texts import footnote_text, intro_text
from data.validate import validate_dataset_format
from src.components import sidebar, table


st.set_page_config(
    page_title='EvalFusion',
    page_icon='../assets/images/logo.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)
sidebar.render_sidebar()

col1, col2 = st.columns([1.3, 1])

col1.image('../assets/images/logo.svg')
col1.title('EvalFusion')
col1.markdown(intro_text)

uploaded_file = col1.file_uploader('Upload your Dataset file here:', type=['json'])
if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)

        if validate_dataset_format(data):
            table.display_table(data)
        else:
            col1.error('The uploaded file does not match the expected format!')
    except Exception as e:
        col1.error(f'An error occurred while processing the file!')


# tab1, tab2, tab3 = st.tabs(["First", "Second", "Third"])
# with tab1:
#     st.write("First")

# with tab2:
#     st.write("Second")

# with tab3:
#     st.write("Third")

# with col1.expander(label="Configuration", icon="⚙️"):
#     st.multiselect("Choose frameworks...", ["RAGAS", "DeepEval", "MLflow", "Phoenix", "TruLens"])
#     st.button("Submit")

st.markdown(footnote_text)
