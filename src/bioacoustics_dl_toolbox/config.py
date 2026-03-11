"""
Typed dataclass configurations for all library components.

Every config is a frozen dataclass: immutable after creation, serializable
via ``dataclasses.asdict()``, and diffable for experiment tracking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SpectrogramConfig:
    """Parameters controlling the audio-to-spectrogram pipeline."""

    sample_rate: int = 44100
    preemphasis: float = 0.98
    n_fft: int = 4096
    hop_length: int = 441
    n_freq_bins: int = 256
    f_min: int = 500
    f_max: int = 10000
    freq_compression: Literal["linear", "mel", "mfcc"] = "linear"
    min_level_db: float = -100
    ref_level_db: float = 20


@dataclass(frozen=True)
class NormalizationConfig:
    """Selects between dB normalization and min-max normalization."""

    mode: Literal["db", "min_max"] = "db"
    min_level_db: float = -100
    ref_level_db: float = 20


@dataclass(frozen=True)
class AugmentationConfig:
    """Data augmentation settings for training."""

    enabled: bool = False
    amplitude_increase_db: int = 3
    amplitude_decrease_db: int = -6
    time_stretch_from: float = 0.5
    time_stretch_to: float = 2.0
    pitch_shift_from: float = 0.5
    pitch_shift_to: float = 1.5
    noise_files: list[str] = field(default_factory=list)
    noise_min_snr: int = 12
    noise_max_snr: int = -3


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the residual encoder."""

    input_channels: int = 1
    conv_kernel_size: int = 7
    max_pool: Literal[0, 1, 2] = 1
    resnet_size: Literal[18, 34, 50, 101, 152] = 18


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for the classification head."""

    input_channels: int = 512
    pooling: Literal["avg", "max"] = "avg"
    num_classes: int = 2


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset construction and splitting."""

    sequence_length_ms: int = 1280
    split_fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    split_seed: int | None = None
    split_per_dir: bool = False
    filter_broken_audio: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the training loop."""

    max_epochs: int = 500
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-5
    beta1: float = 0.5
    lr_patience_epochs: int = 8
    lr_decay_factor: float = 0.5
    early_stopping_patience_epochs: int = 20
    epochs_per_eval: int = 2
    val_metric: str = "accuracy"
    val_metric_mode: Literal["min", "max"] = "max"
    use_cuda: bool = True
    pin_memory: bool = True


def save_config(config: object, path: Path) -> None:
    """Serialize a dataclass config to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_handle:
        json.dump(asdict(config), file_handle, indent=2)  # type: ignore[arg-type]


def load_config_dict(path: Path) -> dict:
    """Load a JSON config file into a plain dict."""
    with open(path) as file_handle:
        return json.load(file_handle)  # type: ignore[no-any-return]
