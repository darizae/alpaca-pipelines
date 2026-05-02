from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class RfFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate_hz: int = 48000
    min_duration_s: float = 0.4
    pad_short_segments: bool = True
    pad_mode: Literal["constant"] = "constant"

    n_fft: int = 2048
    hop_length: int = 1024
    n_mfcc: int = 13
    include_deltas: bool = True
    fmin_hz: float = 0.0
    fmax_hz: float = 4000.0

    @field_validator("sample_rate_hz", "n_fft", "hop_length", "n_mfcc")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0")
        return value

    @field_validator("min_duration_s")
    @classmethod
    def _validate_min_duration(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("min_duration_s must be > 0")
        return value

    @field_validator("fmin_hz")
    @classmethod
    def _validate_fmin(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("fmin_hz must be >= 0")
        return value

    @field_validator("fmax_hz")
    @classmethod
    def _validate_fmax(cls, value: float, info: ValidationInfo) -> float:
        fmin = float(info.data.get("fmin_hz", 0.0))
        if value < fmin:
            raise ValueError("fmax_hz must be >= fmin_hz")
        return value
