from pydantic import BaseModel

from eval_fusion_core.base import EvalFusionBaseEmbeddingModel, EvalFusionBaseLLM


class EvalFusionLLMSettings(BaseModel):
    base_type: type[EvalFusionBaseLLM]
    args: tuple
    kwargs: dict


class EvalFusionEmbeddingModelSettings(BaseModel):
    base_type: type[EvalFusionBaseEmbeddingModel]
    args: tuple
    kwargs: dict
