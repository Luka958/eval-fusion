import streamlit as st

from src.components import sidebar

from eval_fusion_streamlit.constants.texts_main import *


st.set_page_config(
    page_title='Eval Fusion',
    page_icon='../assets/images/logo.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)

sidebar.display_sidebar()

col1, _ = st.columns([1.75, 1])
col1.markdown(intro_text, unsafe_allow_html=True)

col1.markdown(next_button)
if st.button('Next'):
    st.switch_page('pages/1_1️⃣_Upload_Dataset.py')

st.markdown(footnote_text)
