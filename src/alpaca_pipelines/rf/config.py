from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RfFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_fft: int = 2048
    hop_length: int = 1024
    n_mfcc: int = 13
    include_deltas: bool = True
    fmin_hz: float = 0.0
    fmax_hz: float = 4000.0
