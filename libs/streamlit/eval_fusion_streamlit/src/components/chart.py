import altair as alt
import pandas as pd
import streamlit as st


def display_chart(group_results_mean: dict):
    df = pd.DataFrame(group_results_mean)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('Metrics', sort='-y', axis=None),
            y='Average Score',
            color='Metrics',
            tooltip=['Metrics', 'Average Score', 'Group'],
        )
        .properties(width='container')
        .configure_legend(columns=1, labelLimit=1000)
    )

    col11, _ = st.container().columns([2, 1])
    col11.altair_chart(chart, use_container_width=True)
