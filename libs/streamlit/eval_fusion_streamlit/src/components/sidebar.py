import streamlit as st


def display_sidebar():
    with open('../assets/sidebar.css') as f:
        st.markdown(f'<style>{ f.read()}</style>', unsafe_allow_html=True)

    st.sidebar.image('../assets/images/logo.svg')
    st.sidebar.markdown('# Eval Fusion')
