"""
Training run specification.

Defines the complete, serializable configuration for a training run.
The spec is stored in run_state.json and is immutable after creation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpectrogramSpec(BaseModel):
    """Spectrogram pipeline parameters."""

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


class NormalizationSpec(BaseModel):
    """Normalization parameters."""

    mode: Literal["db", "min_max"] = "db"
    min_level_db: float = -100
    ref_level_db: float = 20


class AugmentationSpec(BaseModel):
    """Data augmentation parameters."""

    enabled: bool = False
    amplitude_increase_db: int = 3
    amplitude_decrease_db: int = -6
    time_stretch_from: float = 0.5
    time_stretch_to: float = 2.0
    pitch_shift_from: float = 0.5
    pitch_shift_to: float = 1.5
    noise_files: list[str] = Field(default_factory=list)
    noise_min_snr: int = 12
    noise_max_snr: int = -3


class EncoderSpec(BaseModel):
    """Residual encoder parameters."""

    input_channels: int = 1
    conv_kernel_size: int = 7
    max_pool: Literal[0, 1, 2] = 1
    resnet_size: Literal[18, 34, 50, 101, 152] = 18


class ClassifierSpec(BaseModel):
    """Classification head parameters."""

    input_channels: int = 512
    pooling: Literal["avg", "max"] = "avg"
    num_classes: int = 2


class TrainingHyperparameters(BaseModel):
    """Training loop hyperparameters."""

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


class TrainingRunSpec(BaseModel):
    """Complete specification for a training run.

    This is serialized into run_state.json and is immutable after creation.
    The backend or CLI constructs this, the executor consumes it.
    """

    dataset_name: str
    run_name: str = ""
    sequence_length: int = 128
    spectrogram: SpectrogramSpec = Field(default_factory=SpectrogramSpec)
    normalization: NormalizationSpec = Field(default_factory=NormalizationSpec)
    augmentation: AugmentationSpec = Field(default_factory=AugmentationSpec)
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)
    classifier: ClassifierSpec = Field(default_factory=ClassifierSpec)
    training: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    cache_dir: str | None = None
    positive_class: str = "target"

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for RunState.spec."""
        return self.model_dump()

    @classmethod
    def from_spec_dict(cls, spec: dict[str, Any]) -> TrainingRunSpec:
        """Deserialize from RunState.spec."""
        return cls.model_validate(spec)
