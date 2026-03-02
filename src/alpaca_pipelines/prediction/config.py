"""
Prediction run specification.

Supports two modes:
1. Tape prediction: sliding window over raw audio files
2. Dataset prediction: inference on the test split of a built dataset
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NormalizationSpec(BaseModel):
    """Normalization parameters for prediction."""

    mode: Literal["db", "min_max"] = "db"
    min_level_db: float = -100
    ref_level_db: float = 20


class PredictionRunSpec(BaseModel):
    """Complete specification for a prediction run."""

    model_path: str
    mode: Literal["tape", "dataset"] = "tape"

    audio_files: list[str] = Field(default_factory=list)
    dataset_name: str | None = None

    sequence_length_ms: int = 1280
    hop_ms: int = 640
    batch_size: int = 16
    num_workers: int = 4
    use_cuda: bool = True

    normalization: NormalizationSpec = Field(default_factory=NormalizationSpec)

    detection_threshold: float = 0.5
    merge_overlapping: bool = True
    min_detection_duration_s: float = 0.0

    apply_rf_filter: bool = False
    rf_model_path: str | None = None

    run_name: str = ""

    def to_spec_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_spec_dict(cls, spec: dict[str, Any]) -> PredictionRunSpec:
        return cls.model_validate(spec)
