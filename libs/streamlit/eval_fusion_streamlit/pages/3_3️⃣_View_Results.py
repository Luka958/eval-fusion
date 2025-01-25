import streamlit as st

from eval_fusion_streamlit.constants.page3 import *
from src.components import sidebar


st.set_page_config(
    page_title='View results',
    page_icon='../assets/images/logo.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)

sidebar.display_sidebar()


def handle_start_evaluation():
    st.session_state.results = True


col1, _ = st.columns([1.75, 1])
col1.markdown(view_results_text)

if not 'results' in st.session_state:
    col1.markdown(start_evaluation_text)
    col1.button('Start evaluation', on_click=handle_start_evaluation)
else:
    st.write('RESULTS....')
