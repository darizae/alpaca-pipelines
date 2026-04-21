"""Contracts for prediction manual-review generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class PredictionReviewSpectrogramConfig(BaseModel):
    """Spectrogram rendering configuration for prediction manual review."""

    model_config = ConfigDict(extra="forbid")

    window_function: Literal["hann"] = "hann"
    window_size_samples: int = 2002
    hop_size_samples: int = 1001
    dft_size: int = 2048
    clipping_enabled: bool = False
    clipping_min_db: float = -120.0
    clipping_max_db: float = 0.0
    averaging: int = 1
    auto_apply: bool = False

    colormap: Literal["magma"] = "magma"
    show_axes: bool = True
    x_axis_label: str = "Time (s)"
    y_axis_label: str = "Frequency (kHz)"
    show_colorbar: bool = True

    @model_validator(mode="after")
    def validate_shape(self) -> "PredictionReviewSpectrogramConfig":
        if self.window_size_samples <= 0:
            raise ValueError("window_size_samples must be > 0")
        if self.hop_size_samples <= 0:
            raise ValueError("hop_size_samples must be > 0")
        if self.hop_size_samples > self.window_size_samples:
            raise ValueError("hop_size_samples must be <= window_size_samples")
        if self.dft_size < self.window_size_samples:
            raise ValueError("dft_size must be >= window_size_samples")
        if self.averaging <= 0:
            raise ValueError("averaging must be > 0")
        if self.clipping_min_db >= self.clipping_max_db:
            raise ValueError("clipping_min_db must be < clipping_max_db")
        if not self.x_axis_label.strip():
            raise ValueError("x_axis_label cannot be empty")
        if not self.y_axis_label.strip():
            raise ValueError("y_axis_label cannot be empty")
        return self

    def overlap_percent(self) -> float:
        return round(
            (1.0 - (self.hop_size_samples / self.window_size_samples)) * 100.0,
            6,
        )


class PredictionReviewSessionItem(BaseModel):
    """Single detection item to render for manual review."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    audio_file: str
    start_s: float
    end_s: float
    detection_score: float | None = None
    review_item_id: str | None = None
    detection_index: int | None = None
    source_collection_name: str | None = None
    source_category_dir: str | None = None
    source_relative_path: str | None = None
    source_recording_key: str | None = None
    payload_json: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_item(self) -> "PredictionReviewSessionItem":
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if not self.audio_file.strip():
            raise ValueError("audio_file cannot be empty")
        if self.start_s < 0:
            raise ValueError("start_s must be >= 0")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be > start_s")
        if self.detection_index is not None and self.detection_index < 0:
            raise ValueError("detection_index must be >= 0")
        return self


class PredictionReviewSessionManifest(BaseModel):
    """Session manifest consumed by review artifact generation commands."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    prediction_run_id: str
    session_id: str
    items: list[PredictionReviewSessionItem]
    spectrogram_config: PredictionReviewSpectrogramConfig | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "PredictionReviewSessionManifest":
        if not self.prediction_run_id.strip():
            raise ValueError("prediction_run_id cannot be empty")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not self.items:
            raise ValueError("items must contain at least one entry")

        seen: set[str] = set()
        for item in self.items:
            if item.item_id in seen:
                raise ValueError("Duplicate item_id in manifest: {}".format(item.item_id))
            seen.add(item.item_id)

        return self
