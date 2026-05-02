from __future__ import annotations

from typing import Final

import librosa
import numpy as np
from numpy.typing import NDArray

_ROBUST_FEATURE_NAMES: Final[tuple[str, ...]] = (
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


def _band_slice(frequencies_hz: NDArray[np.float64], fmin: float, fmax: float) -> slice:
    low_index = int(np.searchsorted(frequencies_hz, max(0.0, float(fmin)), side="left"))
    high_index = int(np.searchsorted(frequencies_hz, float(fmax), side="right"))
    low_index = max(0, min(low_index, int(frequencies_hz.shape[0])))
    high_index = max(low_index + 1, min(high_index, int(frequencies_hz.shape[0])))
    return slice(low_index, high_index)


def _quantile_from_pdf(
    pdf: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    quantile: float,
) -> float:
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
    weight_current = (quantile - previous_cumulative) / (
        current_cumulative - previous_cumulative + 1e-12
    )
    return float(
        (1.0 - weight_current) * coordinates[index - 1] + weight_current * coordinates[index]
    )


def _entropy_bits(probability: NDArray[np.float64]) -> float:
    clipped = np.clip(probability, 1e-12, 1.0)
    return float(-(clipped * np.log2(clipped)).sum())


def raven_robust_features(
    y: NDArray[np.float32],
    sr: int,
    fmin: float,
    fmax: float,
    n_fft: int = 2048,
    hop_length: int = 1024,
    window: str = "hann",
    center: bool = True,
) -> dict[str, float]:
    if y.size == 0:
        return {key: float("nan") for key in _ROBUST_FEATURE_NAMES}

    stft_complex = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )
    power = (np.abs(stft_complex) ** 2).astype(np.float64)
    frequencies_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float64)
    times_s = librosa.frames_to_time(
        np.arange(power.shape[1]),
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
    ).astype(np.float64)

    freq_slice = _band_slice(frequencies_hz, fmin=fmin, fmax=fmax)
    band_power = power[freq_slice, :]

    time_envelope = band_power.sum(axis=0)
    time_5 = _quantile_from_pdf(time_envelope, times_s, 0.05)
    time_25 = _quantile_from_pdf(time_envelope, times_s, 0.25)
    time_75 = _quantile_from_pdf(time_envelope, times_s, 0.75)
    time_95 = _quantile_from_pdf(time_envelope, times_s, 0.95)
    duration_90 = max(0.0, time_95 - time_5)
    duration_50 = max(0.0, time_75 - time_25)

    spectrum_mean = band_power.mean(axis=1)
    band_frequencies = frequencies_hz[freq_slice]
    freq_5 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.05)
    freq_25 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.25)
    freq_50 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.50)
    freq_75 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.75)
    freq_95 = _quantile_from_pdf(spectrum_mean, band_frequencies, 0.95)
    bandwidth_50 = max(0.0, freq_75 - freq_25)
    bandwidth_90 = max(0.0, freq_95 - freq_5)

    power_per_time = band_power.sum(axis=0) + 1e-12
    probability_frequency_given_time = band_power / power_per_time
    entropy_per_time = -(
        np.clip(probability_frequency_given_time, 1e-12, 1.0)
        * np.log2(np.clip(probability_frequency_given_time, 1e-12, 1.0))
    ).sum(axis=0)
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
