from eval_fusion_core.models import EvaluationInput


tmp = EvaluationInput(input='a', output='b', ground_truth='', relevant_chunks=[])
print(tmp.input)
