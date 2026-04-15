from __future__ import annotations

import librosa
import numpy as np
from numpy.typing import NDArray


def _slice_signal(y: NDArray[np.float32], sr: int, t0: float, t1: float) -> NDArray[np.float32]:
    start_index = max(0, int(round(float(t0) * sr)))
    end_index = min(int(y.shape[0]), int(round(float(t1) * sr)))
    if end_index <= start_index:
        return np.zeros(0, dtype=np.float32)
    return y[start_index:end_index]


def _summarize_matrix(matrix: NDArray[np.float64], prefix: str) -> dict[str, float]:
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    summary: dict[str, float] = {}
    for index in range(int(matrix.shape[0])):
        coefficient = index + 1
        summary[f"{prefix}{coefficient}_mean"] = float(means[index])
        summary[f"{prefix}{coefficient}_std"] = float(stds[index])
    return summary


def mfcc_summary(
    y: NDArray[np.float32],
    sr: int,
    t0: float,
    t1: float,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 1024,
    include_deltas: bool = True,
) -> dict[str, float]:
    segment = _slice_signal(y, sr, t0, t1)
    if segment.size == 0:
        feature_names: list[str] = []
        for index in range(1, n_mfcc + 1):
            feature_names.extend([f"mfcc{index}_mean", f"mfcc{index}_std"])
        if include_deltas:
            for index in range(1, n_mfcc + 1):
                feature_names.extend([f"d_mfcc{index}_mean", f"d_mfcc{index}_std"])
            for index in range(1, n_mfcc + 1):
                feature_names.extend([f"dd_mfcc{index}_mean", f"dd_mfcc{index}_std"])
        return {name: float("nan") for name in feature_names}

    mfcc_matrix = librosa.feature.mfcc(
        y=segment.astype(np.float32, copy=False),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float64)

    summary = _summarize_matrix(mfcc_matrix, "mfcc")

    if include_deltas:
        delta_1 = librosa.feature.delta(mfcc_matrix, order=1).astype(np.float64)
        delta_2 = librosa.feature.delta(mfcc_matrix, order=2).astype(np.float64)
        summary.update(_summarize_matrix(delta_1, "d_mfcc"))
        summary.update(_summarize_matrix(delta_2, "dd_mfcc"))

    return summary
