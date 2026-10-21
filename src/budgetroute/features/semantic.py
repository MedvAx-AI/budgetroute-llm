"""One pinned prompt-only feature path shared by fitting and online routing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from budgetroute.exceptions import RouterTrainingError
from budgetroute.retrieval.embeddings import TransformersEmbedder

PREPROCESSING_IDENTITY = "mean_pool_attention_mask_l2_prompt_only_v1"


class SemanticFeatureSpec(BaseModel):
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = Field(
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", pattern=r"^[a-f0-9]{40}$"
    )
    device: Literal["cpu", "cuda"] = "cpu"
    max_input_tokens: int = Field(default=256, ge=1, le=4096)


class SemanticFeatures:
    def __init__(self, spec: SemanticFeatureSpec) -> None:
        self.spec = spec
        self.embedder = TransformersEmbedder(
            spec.model_id, spec.device, spec.revision, max_input_tokens=spec.max_input_tokens
        )

    @property
    def truncation_count(self) -> int:
        return self.embedder.truncation_count

    def transform(self, prompts: list[str]) -> np.ndarray:
        matrix = self.embedder.embed(prompts)
        if matrix.ndim != 2 or matrix.shape[0] != len(prompts) or not np.isfinite(matrix).all():
            raise RouterTrainingError("invalid semantic feature matrix")
        return matrix

    def close(self) -> None:
        self.embedder.cleanup()
