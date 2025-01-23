import pandas as pd
import streamlit as st


def display_table(data: list):
    dataframe = pd.DataFrame(data)
    # st.table(dataframe.head(2))
    # a = st.data_editor(dataframe)
    st.dataframe(dataframe)
