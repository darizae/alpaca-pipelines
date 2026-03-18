"""
Prediction run specification.

Supports two modes:
1. Tape prediction: sliding window over raw audio files
2. Dataset prediction: inference on the test split of a built dataset
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class NormalizationSpec(BaseModel):
    """Normalization parameters for prediction."""

    mode: Literal["db", "min_max"] = "db"
    min_level_db: float = -100
    ref_level_db: float = 20


class PredictionRunSpec(BaseModel):
    """Complete specification for a prediction run."""

    model_path: str
    mode: Literal["tape", "dataset", "collection"] = "tape"

    audio_files: list[str] = Field(default_factory=list)
    dataset_name: str | None = None
    collection_names: list[str] = Field(default_factory=list)
    source_category_dirs: list[str] = Field(default_factory=lambda: ["raw_recordings"])

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

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> "PredictionRunSpec":
        if self.mode == "dataset":
            if not self.dataset_name:
                raise ValueError("dataset_name is required for dataset mode")
            if self.audio_files:
                raise ValueError("audio_files must be empty for dataset mode")
            if self.collection_names:
                raise ValueError("collection_names must be empty for dataset mode")
            return self

        if self.mode == "tape":
            if not self.audio_files:
                raise ValueError("audio_files are required for tape mode")
            if self.dataset_name is not None:
                raise ValueError("dataset_name must be empty for tape mode")
            if self.collection_names:
                raise ValueError("collection_names must be empty for tape mode")
            return self

        if not self.collection_names:
            raise ValueError("collection_names are required for collection mode")
        if self.audio_files:
            raise ValueError("audio_files must be empty for collection mode")
        if self.dataset_name is not None:
            raise ValueError("dataset_name must be empty for collection mode")
        if not self.source_category_dirs:
            raise ValueError("source_category_dirs are required for collection mode")

        for collection_name in self.collection_names:
            if not collection_name.startswith("audio_collection_"):
                raise ValueError("collection_names entries must start with 'audio_collection_'")
            if not _is_safe_path_segment(collection_name):
                raise ValueError("collection_names entries must be safe path segments")
        for category_dir in self.source_category_dirs:
            if not _is_safe_path_segment(category_dir):
                raise ValueError("source_category_dirs entries must be safe path segments")
        return self

    def to_spec_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_spec_dict(cls, spec: dict[str, Any]) -> PredictionRunSpec:
        return cls.model_validate(spec)


def _is_safe_path_segment(value: str) -> bool:
    if not value or value in {".", ".."}:
        return False
    return "/" not in value and "\\" not in value and ".." not in value
