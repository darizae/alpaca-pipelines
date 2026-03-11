from __future__ import annotations

import math
from typing import Final, Mapping

import librosa
import numpy as np
from numpy.typing import NDArray

from bioacoustics_dl_toolbox.rf.types import FeatureValues

_ROBUST_SCHEMA: Final[tuple[str, ...]] = (
    "Dur 90% (s)",
    "Dur 50% (s)",
    "Center Freq (Hz)",
    "Freq 5% (Hz)",
    "Freq 25% (Hz)",
    "Freq 75% (Hz)",
    "Freq 95% (Hz)",
    "BW 50% (Hz)",
    "BW 90% (Hz)",
    "Avg Entropy (bits)",
    "Agg Entropy (bits)",
)


def _slice_signal(signal: NDArray[np.float32], sample_rate: int, start_s: float, end_s: float) -> NDArray[np.float32]:
    start_index = max(0, int(round(start_s * sample_rate)))
    end_index = min(int(signal.shape[0]), int(round(end_s * sample_rate)))
    if end_index <= start_index:
        return np.zeros(0, dtype=np.float32)
    return signal[start_index:end_index]


def _band_slice(frequencies_hz: NDArray[np.float64], low_hz: float | None, high_hz: float | None) -> slice:
    low_index = 0 if low_hz is None else int(np.searchsorted(frequencies_hz, max(0.0, float(low_hz)), side="left"))
    if high_hz is None or high_hz <= 0:
        high_index = int(frequencies_hz.shape[0])
    else:
        high_index = int(np.searchsorted(frequencies_hz, float(high_hz), side="right"))

    low_index = max(0, min(low_index, int(frequencies_hz.shape[0])))
    high_index = max(low_index + 1, min(high_index, int(frequencies_hz.shape[0])))
    return slice(low_index, high_index)


def _quantile_from_pdf(pdf: NDArray[np.float64], coordinates: NDArray[np.float64], quantile: float) -> float:
    clipped = np.clip(pdf, 0.0, np.inf)
    total = float(clipped.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float(coordinates[int(coordinates.shape[0] // 2)])

    probability = clipped / total
    cumulative = np.cumsum(probability)
    index = int(np.searchsorted(cumulative, quantile, side="left"))

    if index <= 0:
        return float(coordinates[0])
    if index >= int(coordinates.shape[0]):
        return float(coordinates[-1])

    previous_cumulative = float(cumulative[index - 1])
    current_cumulative = float(cumulative[index])
    weight_current = (quantile - previous_cumulative) / (current_cumulative - previous_cumulative + 1e-12)
    return float((1.0 - weight_current) * coordinates[index - 1] + weight_current * coordinates[index])


def _entropy_bits(probability: NDArray[np.float64]) -> float:
    clipped = np.clip(probability, 1e-12, 1.0)
    return float(-(clipped * np.log2(clipped)).sum())


def raven_robust_features(
    signal: NDArray[np.float32],
    sample_rate: int,
    start_s: float,
    end_s: float,
    low_hz: float | None = None,
    high_hz: float | None = None,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    center: bool = True,
) -> dict[str, float]:
    """
    Compute Raven-style robust selection measurements from an audio segment.

    Returns a dict with a stable schema matching _ROBUST_SCHEMA.
    """
    segment = _slice_signal(signal, sample_rate, start_s, end_s)
    if segment.size == 0:
        return {key: float("nan") for key in _ROBUST_SCHEMA}

    stft_complex = librosa.stft(
        segment,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )
    power_spectral_density = (np.abs(stft_complex) ** 2).astype(np.float64)
    frequencies_hz = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float64)
    times_s = librosa.frames_to_time(
        np.arange(power_spectral_density.shape[1]),
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
    ).astype(np.float64)

    frequency_slice = _band_slice(frequencies_hz, low_hz, high_hz)
    band_power = power_spectral_density[frequency_slice, :]

    time_envelope = band_power.sum(axis=0)
    time_5 = _quantile_from_pdf(time_envelope, times_s, 0.05)
    time_25 = _quantile_from_pdf(time_envelope, times_s, 0.25)
    time_75 = _quantile_from_pdf(time_envelope, times_s, 0.75)
    time_95 = _quantile_from_pdf(time_envelope, times_s, 0.95)
    duration_90 = max(0.0, time_95 - time_5)
    duration_50 = max(0.0, time_75 - time_25)

    spectrum_mean = band_power.mean(axis=1)
    band_frequencies = frequencies_hz[frequency_slice]
    freq_5 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.05)
    freq_25 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.25)
    freq_50 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.50)
    freq_75 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.75)
    freq_95 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.95)
    bandwidth_50 = max(0.0, freq_75 - freq_25)
    bandwidth_90 = max(0.0, freq_95 - freq_5)

    power_per_time = band_power.sum(axis=0) + 1e-12
    probability_frequency_given_time = band_power / power_per_time
    clipped = np.clip(probability_frequency_given_time, 1e-12, 1.0)
    entropy_per_time = -(clipped * np.log2(clipped)).sum(axis=0)
    average_entropy = float(np.mean(entropy_per_time))

    spectrum_probability = spectrum_mean / (float(spectrum_mean.sum()) + 1e-12)
    aggregate_entropy = _entropy_bits(spectrum_probability)

    return {
        "Dur 90% (s)": float(duration_90),
        "Dur 50% (s)": float(duration_50),
        "Center Freq (Hz)": float(freq_50),
        "Freq 5% (Hz)": float(freq_5),
        "Freq 25% (Hz)": float(freq_25),
        "Freq 75% (Hz)": float(freq_75),
        "Freq 95% (Hz)": float(freq_95),
        "BW 50% (Hz)": float(bandwidth_50),
        "BW 90% (Hz)": float(bandwidth_90),
        "Avg Entropy (bits)": float(average_entropy),
        "Agg Entropy (bits)": float(aggregate_entropy),
    }


def mfcc_feature_schema(n_mfcc: int, include_deltas: bool = True) -> list[str]:
    feature_names: list[str] = []
    for coefficient_index in range(1, n_mfcc + 1):
        feature_names.append("MFCC {:02d} mean".format(coefficient_index))
        feature_names.append("MFCC {:02d} std".format(coefficient_index))

    if include_deltas:
        for coefficient_index in range(1, n_mfcc + 1):
            feature_names.append("ΔMFCC {:02d} mean".format(coefficient_index))
            feature_names.append("ΔMFCC {:02d} std".format(coefficient_index))
        for coefficient_index in range(1, n_mfcc + 1):
            feature_names.append("ΔΔMFCC {:02d} mean".format(coefficient_index))
            feature_names.append("ΔΔMFCC {:02d} std".format(coefficient_index))

    return feature_names


def _summarize_feature_matrix(
    matrix: NDArray[np.float64],
    feature_prefix: str,
) -> dict[str, float]:
    if matrix.size == 0:
        raise ValueError("Cannot summarize an empty feature matrix")

    means = matrix.mean(axis=1)
    standard_deviations = matrix.std(axis=1)

    features: dict[str, float] = {}
    for row_index in range(int(matrix.shape[0])):
        coefficient_index = row_index + 1
        features["{} {:02d} mean".format(feature_prefix, coefficient_index)] = float(means[row_index])
        features["{} {:02d} std".format(feature_prefix, coefficient_index)] = float(standard_deviations[row_index])
    return features


def mfcc_summary(
    signal: NDArray[np.float32],
    sample_rate: int,
    start_s: float,
    end_s: float,
    *,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_deltas: bool = True,
) -> dict[str, float]:
    """
    Compute MFCC summary statistics for an audio segment.

    Returned keys are stable and deterministic, matching mfcc_feature_schema().
    """
    segment = _slice_signal(signal, sample_rate, start_s, end_s)
    if segment.size == 0:
        schema = mfcc_feature_schema(n_mfcc=n_mfcc, include_deltas=include_deltas)
        return {key: float("nan") for key in schema}

    mfcc_matrix = librosa.feature.mfcc(
        y=segment.astype(np.float32, copy=False),
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float64)

    summary = _summarize_feature_matrix(mfcc_matrix, "MFCC")

    if include_deltas:
        delta_1 = librosa.feature.delta(mfcc_matrix, order=1).astype(np.float64)
        delta_2 = librosa.feature.delta(mfcc_matrix, order=2).astype(np.float64)
        summary.update(_summarize_feature_matrix(delta_1, "ΔMFCC"))
        summary.update(_summarize_feature_matrix(delta_2, "ΔΔMFCC"))

    return summary


def compute_rf_features(
    signal: NDArray[np.float32],
    sample_rate: int,
    start_s: float,
    end_s: float,
    *,
    low_hz: float | None = None,
    high_hz: float | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
    include_deltas: bool = True,
    window: str = "hann",
    center: bool = True,
) -> dict[str, float]:
    """
    Compute the combined RF feature set (robust + MFCC summaries).
    """
    robust = raven_robust_features(
        signal=signal,
        sample_rate=sample_rate,
        start_s=start_s,
        end_s=end_s,
        low_hz=low_hz,
        high_hz=high_hz,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )
    mfcc = mfcc_summary(
        signal=signal,
        sample_rate=sample_rate,
        start_s=start_s,
        end_s=end_s,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        include_deltas=include_deltas,
    )
    return {**robust, **mfcc}


def align_features_to_schema(
    features: Mapping[str, float] | FeatureValues,
    schema: list[str] | tuple[str, ...],
) -> NDArray[np.float64]:
    """
    Align a feature dict to an explicit schema order.

    No fallbacks:
    - Missing keys raise KeyError
    - Non-finite values are allowed (caller decides), but the vectorization is strict.
    """
    values: list[float] = []
    for key in schema:
        if key not in features:
            raise KeyError("Missing required feature: {}".format(key))
        values.append(float(features[key]))
    return np.asarray(values, dtype=np.float64).reshape(1, -1)
