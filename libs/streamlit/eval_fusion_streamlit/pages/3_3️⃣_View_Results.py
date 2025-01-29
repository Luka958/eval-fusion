import altair as alt
import pandas as pd
import streamlit as st

from eval_fusion_streamlit.constants.const import options, semantic_groups
from eval_fusion_streamlit.constants.texts_page3 import *
from src.components import chart, sidebar
from src.evaluation import evaluate


st.set_page_config(
    page_title='View results',
    page_icon='../assets/images/logo.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)

sidebar.display_sidebar()


def handle_start_evaluation():
    for framework, metrics in st.session_state.selected_metrics.items():
        st.session_state[framework] = evaluate(framework, metrics)

    st.session_state.has_results = True


col1, _ = st.columns([1.75, 1])
start_evaluation_disabled = False
col1.markdown(view_results_text)

if 'metrics_option' not in st.session_state:
    start_evaluation_disabled = True
    col1.error(
        'No metrics have been selected for evaluation. Please, follow the steps from the beginning.'
    )

if not 'has_results' in st.session_state or not st.session_state.has_results:
    col1.markdown(start_evaluation_text)
    col1.button(
        label='Start evaluation',
        on_click=handle_start_evaluation,
        disabled=start_evaluation_disabled,
    )
else:
    if options[st.session_state.metrics_option] == 1:
        for selected_group in st.session_state.selected_groups:
            group_results = {}
            group_results_mean = []

            for framework, metrics in semantic_groups[selected_group].items():
                for metric in metrics:
                    scores = st.session_state[framework][metric]
                    length = len(scores)

                    if 'ID' not in group_results:
                        group_results['ID'] = list(range(1, length + 1))

                    group_results[f'{metric} [{framework}]'] = scores
                    avg_score = round(sum(scores) / length, 3)

                    group_results_mean.append(
                        {
                            'Metrics': f'{metric} [{framework}]',
                            'Average Score': avg_score,
                            'Group': selected_group,
                        }
                    )

            with st.expander(selected_group):
                st.dataframe(group_results)
                chart.display_chart(group_results_mean)
    else:
        for framework in st.session_state.selected_metrics:
            with st.expander(framework):
                st.dataframe(st.session_state[framework])
