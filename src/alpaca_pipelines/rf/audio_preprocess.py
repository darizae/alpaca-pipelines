from __future__ import annotations

from typing import cast

import librosa
import numpy as np
from numpy.typing import NDArray

from alpaca_pipelines.rf.config import RfFeatureConfig


def to_mono(signal: NDArray[np.float32]) -> NDArray[np.float32]:
    if signal.ndim == 1:
        return signal.astype(np.float32, copy=False)
    if signal.ndim != 2:
        raise ValueError("signal must be 1D or 2D")
    if signal.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    mono = np.mean(signal, axis=1, dtype=np.float32)
    return cast(NDArray[np.float32], np.asarray(mono, dtype=np.float32))


def resample_if_needed(
    signal: NDArray[np.float32],
    source_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    if source_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    resampled = librosa.resample(
        signal.astype(np.float32, copy=False),
        orig_sr=source_sr,
        target_sr=target_sr,
    )
    return cast(NDArray[np.float32], np.asarray(resampled, dtype=np.float32))


def slice_seconds(
    signal: NDArray[np.float32],
    sr: int,
    t0: float,
    t1: float,
) -> NDArray[np.float32]:
    n_samples = int(signal.shape[0])
    start = int(round(max(0.0, float(t0)) * float(sr)))
    end = int(round(max(0.0, float(t1)) * float(sr)))
    start = min(max(0, start), n_samples)
    end = min(max(0, end), n_samples)
    if end <= start:
        return np.zeros(0, dtype=np.float32)
    return signal[start:end].astype(np.float32, copy=False)


def pad_to_min_duration(
    segment: NDArray[np.float32],
    sr: int,
    min_duration_s: float,
    pad_short_segments: bool,
) -> NDArray[np.float32]:
    if not pad_short_segments:
        return segment.astype(np.float32, copy=False)

    min_samples = int(round(float(sr) * float(min_duration_s)))
    if min_samples <= 0:
        return segment.astype(np.float32, copy=False)
    if segment.size == 0:
        return np.zeros(min_samples, dtype=np.float32)
    if int(segment.shape[0]) >= min_samples:
        return segment.astype(np.float32, copy=False)

    total_pad = min_samples - int(segment.shape[0])
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return np.pad(segment, (left_pad, right_pad), mode="constant").astype(np.float32, copy=False)


def prepare_rf_segment(
    signal: NDArray[np.float32],
    source_sr: int,
    t0: float,
    t1: float,
    config: RfFeatureConfig,
) -> tuple[NDArray[np.float32], int]:
    mono = to_mono(signal)
    resampled = resample_if_needed(
        signal=mono,
        source_sr=source_sr,
        target_sr=config.sample_rate_hz,
    )
    segment = slice_seconds(
        signal=resampled,
        sr=config.sample_rate_hz,
        t0=t0,
        t1=t1,
    )
    padded = pad_to_min_duration(
        segment=segment,
        sr=config.sample_rate_hz,
        min_duration_s=config.min_duration_s,
        pad_short_segments=config.pad_short_segments,
    )
    return padded.astype(np.float32, copy=False), config.sample_rate_hz
