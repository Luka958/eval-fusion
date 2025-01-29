import json

from copy import deepcopy

import streamlit as st

from eval_fusion_core.models.output import EvaluationOutput, EvaluationOutputEntry


def evaluate(to_del, metrics: list[str]) -> dict:
    # TODO: evaluate...
    # DATASET => st.session_state.dataset

    ### MOCK BLOCK start ###
    name = {
        'RAGAS': 'ragas.json',
        'DeepEval': 'deepeval.json',
        'MLflow': 'mlflow.json',
        'Phoenix Arize AI': 'phoenix.json',
        'TruLens': 'trulens.json',
    }
    with open(
        f'D:\\FER\\diplomski\\3. semestar\\OPJ\\old\\{name[to_del]}', 'r'
    ) as file:
        data = json.load(file)

    results = []
    for record in data:
        id, input_id, output_entries_raw = record.values()
        output_entries = []
        for output_entry_raw in output_entries_raw:
            metric_name, score, reason = output_entry_raw.values()
            if metric_name in metrics:
                output_entries.append(
                    EvaluationOutputEntry(
                        metric_name=metric_name, score=float(score), reason=reason
                    )
                )

        results.append(
            EvaluationOutput(id=id, input_id=input_id, output_entries=output_entries)
        )

    for _ in range(50):
        results.append(deepcopy(results[0]))
    ### MOCK BLOCK end ###

    final_results = {'ID': []}
    final_results.update({key: [] for key in metrics})

    for idx, evaluation_output in enumerate(results):
        final_results['ID'].append(idx + 1)
        for evaluation_output_entry in evaluation_output.output_entries:
            metric_name = evaluation_output_entry.metric_name
            score = evaluation_output_entry.score

            final_results[metric_name].append(score)

    return final_results
