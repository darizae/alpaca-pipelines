"""
RF training run specification.

Defines the complete, serializable configuration for training an RF
filter model used in post-processing of CNN detections.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from alpaca_pipelines.rf.config import RfFeatureConfig


class RfTrainingRunSpec(BaseModel):
    """Complete specification for an RF training run."""

    dataset_name: str
    run_name: str = ""

    positive_class: str = "target"

    random_state: int = 1337

    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["sqrt", "log2"] | float | int | None = "sqrt"
    class_weight: Literal["balanced", "balanced_subsample"] | dict[str, float] | None = "balanced"
    n_jobs: int = -1
    feature_config: RfFeatureConfig = Field(default_factory=RfFeatureConfig)

    def to_spec_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_spec_dict(cls, spec: dict[str, Any]) -> RfTrainingRunSpec:
        return cls.model_validate(spec)
