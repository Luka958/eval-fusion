import streamlit as st

from eval_fusion_streamlit.constants.const import (
    framework_metrics,
    options,
    semantic_groups,
)
from eval_fusion_streamlit.constants.texts_page2 import *
from src.components import sidebar


st.set_page_config(
    page_title='Choose metrics',
    page_icon='../assets/images/logo.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)

sidebar.display_sidebar()
st.session_state.has_results = False


def handle_selected_groups(selected_groups: list[str]) -> dict:
    selected_metrics = {}
    for group in selected_groups:
        for framework, metrics in semantic_groups[group].items():
            if framework not in selected_metrics:
                selected_metrics[framework] = []
            selected_metrics[framework].extend(metrics)

    for framework in selected_metrics:
        selected_metrics[framework] = list(set(selected_metrics[framework]))

    return selected_metrics


col1, _ = st.columns([1.75, 1])
col1.markdown(choose_metrics_text)

if 'metrics_option' not in st.session_state:
    st.session_state.metrics_option = list(options.keys())[0]

metrics_option = col1.radio(
    label='How would you like to select metrics?',
    options=options.keys(),
    index=options[st.session_state.metrics_option],
)

if metrics_option != st.session_state.metrics_option:
    st.session_state.metrics_option = metrics_option


if options[st.session_state.metrics_option] == 0:
    st.session_state.selected_metrics = framework_metrics

elif options[st.session_state.metrics_option] == 1:
    col1.markdown(semantic_groups_text)
    selected_groups = col1.multiselect(
        label='Choose one or more semantic groups:', options=semantic_groups.keys()
    )
    st.session_state.selected_groups = selected_groups
    st.session_state.selected_metrics = handle_selected_groups(selected_groups)

elif options[st.session_state.metrics_option] == 2:
    col1.markdown(frameworks_metrics_text)
    selected_frameworks = col1.multiselect(
        label='Choose one or more frameworks:', options=framework_metrics.keys()
    )

    if selected_frameworks:
        col1.divider()
        selected_metrics = {}

        for framework in selected_frameworks:
            with col1.expander(f'Metrics for {framework}', expanded=True):
                selected_metrics[framework] = st.multiselect(
                    label=f'Choose metrics for {framework}:',
                    options=framework_metrics[framework],
                )

        st.session_state.selected_metrics = selected_metrics

col1.markdown(next_button)
if st.button('Next'):
    st.switch_page('pages/3_3️⃣_View_Results.py')
