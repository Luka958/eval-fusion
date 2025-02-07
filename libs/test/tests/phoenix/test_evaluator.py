import pytest

from eval_fusion_core.enums import MetricTag
from eval_fusion_core.utils.loaders import load_evaluation_inputs
from eval_fusion_phoenix.evaluator import PhoenixEvaluator
from eval_fusion_test.settings import get_openai_settings


@pytest.mark.parametrize('input_count', [1])
def test_evaluator(input_count: int):
    llm_settings, _ = get_openai_settings()
    inputs = load_evaluation_inputs('assets/amnesty_qa.json')

    inputs = inputs[:input_count]
    with PhoenixEvaluator(llm_settings) as evaluator:
        outputs = evaluator.evaluate_by_tag(inputs, MetricTag.ALL)

    for output in outputs:
        for output_entry in output.output_entries:
            print(output_entry, end='\n\n')
