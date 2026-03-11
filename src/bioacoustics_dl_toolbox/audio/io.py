"""
Audio file loading utilities.

Ported from ANIMAL-SPOT ``data/transforms.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import numpy as np
import resampy
import soundfile as sf
import torch


def load_audio_file(
    file_name: str,
    sample_rate: int | None = None,
    mono: bool = True,
) -> torch.Tensor:
    """Load an audio file and return a float tensor of shape ``(channels, samples)``.

    Parameters
    ----------
    file_name:
        Path to the audio file (any format supported by ``soundfile``).
    sample_rate:
        Target sample rate. If provided and different from the file's native
        rate, the audio is resampled using a Kaiser-best filter.
    mono:
        If ``True``, multi-channel audio is averaged to mono.
    """
    audio_data, native_sample_rate = sf.read(file_name, always_2d=True, dtype="float32")
    if mono and audio_data.ndim == 2 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1, keepdims=True)
    if sample_rate is not None and sample_rate != native_sample_rate:
        audio_data = resampy.resample(
            audio_data, native_sample_rate, sample_rate, axis=0, filter="kaiser_best"
        )
    return torch.from_numpy(audio_data).float().t()
