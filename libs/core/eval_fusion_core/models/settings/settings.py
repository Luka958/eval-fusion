from pydantic import BaseModel

from eval_fusion_core.base import EvalFusionBaseEM, EvalFusionBaseLLM


class EvalFusionLLMSettings(BaseModel):
    base_type: type[EvalFusionBaseLLM]
    args: tuple = ()
    kwargs: dict = {}


class EvalFusionEMSettings(BaseModel):
    base_type: type[EvalFusionBaseEM]
    args: tuple = ()
    kwargs: dict = {}
