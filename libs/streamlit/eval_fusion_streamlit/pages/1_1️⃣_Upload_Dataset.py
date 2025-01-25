import json

import streamlit as st

from data.validate import validate_dataset_format
from eval_fusion_streamlit.constants.page1 import *
from src.components import record_container, sidebar


st.set_page_config(
    page_title='Upload Dataset',
    page_icon='../assets/images/logo.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)

sidebar.display_sidebar()


def handle_input_change(ids_input: str):
    ids = [int(id.strip()) for id in ids_input.split(',') if id.strip().isdigit()]
    for id in ids:
        if id < 1 or id > len(st.session_state.dataset):
            ids.remove(id)

    st.session_state['ids_input'] = sorted(set(ids))


def render_records_by_ids(*ids: int):
    for id in ids:
        record = st.session_state.dataset[id - 1]
        with col1.expander(f'Record {id}'):
            record_container.display_record_container(record)


col1, _ = st.columns([1.75, 1])
col1.markdown(prerequisites_text)
st.session_state.openai_api_key = col1.text_input('Enter your OpenAI API key:')
st.session_state.llm_model = col1.selectbox('Select the LLM model', openai_llm_models)
st.session_state.embedding_model = col1.selectbox(
    'Select the Embedding model', openai_embedding_models
)

col1.markdown(upload_dataset_text)
uploaded_file = col1.file_uploader('Upload your Dataset file here:', type=['json'])

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)

        if not validate_dataset_format(data):
            st.session_state.pop('dataset', None)
            st.session_state.pop('ids_input', None)
            col1.error(
                'Dataset does not match the expected format or contains fewer than 5 records!'
            )
        else:
            for i in range(len(data)):
                data[i]['id'] = i + 1
            st.session_state.dataset = data

    except Exception as e:
        col1.error(f'An error occurred while processing the file!')

if 'dataset' in st.session_state:
    col1.markdown(preview_text)
    render_records_by_ids(1, 2)
    col1.markdown('...')
    render_records_by_ids(len(st.session_state.dataset))

    col1.markdown(inspect_text)
    ids_input = col1.text_input(
        'Enter the IDs of the records you want to check, separated by commas:'
    )
    if ids_input:
        handle_input_change(ids_input)

if 'ids_input' in st.session_state:
    render_records_by_ids(*st.session_state.ids_input)

col1.markdown(next_button)
if st.button('Next'):
    st.switch_page('pages/2_2️⃣_Choose_metrics.py')
