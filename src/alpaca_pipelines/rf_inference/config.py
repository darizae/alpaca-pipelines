"""Standalone RF inference run specification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from alpaca_pipelines.rf.config import RfFeatureConfig


class RfInferenceRunSpec(BaseModel):
    """Specification for applying RF filtering to an existing prediction run."""

    model_config = ConfigDict(extra="forbid")

    source_prediction_run_id: str
    rf_training_run_id: str
    source_predictions_dir: str
    rf_model_path: str
    rf_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    rf_feature_config: RfFeatureConfig = Field(default_factory=RfFeatureConfig)
    run_name: str = ""
