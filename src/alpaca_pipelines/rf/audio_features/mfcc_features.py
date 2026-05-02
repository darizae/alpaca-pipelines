from __future__ import annotations

import librosa
import numpy as np
from librosa.util.exceptions import ParameterError
from numpy.typing import NDArray


def _summarize_matrix(matrix: NDArray[np.float64], prefix: str) -> dict[str, float]:
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    summary: dict[str, float] = {}
    for index in range(int(matrix.shape[0])):
        coefficient = index + 1
        summary[f"{prefix}{coefficient}_mean"] = float(means[index])
        summary[f"{prefix}{coefficient}_std"] = float(stds[index])
    return summary


def _delta_feature_names(n_mfcc: int) -> list[str]:
    names: list[str] = []
    for index in range(1, n_mfcc + 1):
        names.extend([f"d_mfcc{index}_mean", f"d_mfcc{index}_std"])
    for index in range(1, n_mfcc + 1):
        names.extend([f"dd_mfcc{index}_mean", f"dd_mfcc{index}_std"])
    return names


def mfcc_summary(
    y: NDArray[np.float32],
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 1024,
    include_deltas: bool = True,
) -> dict[str, float]:
    if y.size == 0:
        raise ValueError("MFCC segment is empty; preprocessing must provide a non-empty segment")

    mfcc_matrix = librosa.feature.mfcc(
        y=y.astype(np.float32, copy=False),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float64)

    summary = _summarize_matrix(mfcc_matrix, "mfcc")

    if include_deltas:
        try:
            delta_1 = librosa.feature.delta(mfcc_matrix, order=1).astype(np.float64)
            delta_2 = librosa.feature.delta(mfcc_matrix, order=2).astype(np.float64)
            summary.update(_summarize_matrix(delta_1, "d_mfcc"))
            summary.update(_summarize_matrix(delta_2, "dd_mfcc"))
        except (ParameterError, ValueError) as exc:
            n_frames = int(mfcc_matrix.shape[1]) if mfcc_matrix.ndim == 2 else 0
            raise ValueError(
                "MFCC delta computation failed: segment_samples={} sample_rate={} n_fft={} "
                "hop_length={} mfcc_frames={}".format(
                    int(y.shape[0]),
                    int(sr),
                    int(n_fft),
                    int(hop_length),
                    n_frames,
                )
            ) from exc

    return summary
