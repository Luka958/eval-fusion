from eval_fusion_core.base import EvalFusionBaseLLM


class MlFlowProxyLLM:
    def __init__(self, llm_delegate: EvalFusionBaseLLM):
        self.llm_delegate = llm_delegate

        # TODO api key in os env?

    def get_model(self) -> str:
        return self.llm_delegate.get_name()
