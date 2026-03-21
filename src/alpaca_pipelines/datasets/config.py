from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NoiseMiningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts_per_slot: int = Field(
        default=20,
        description="Maximum attempts before giving up on a single noise slot.",
    )
    source_category_dirs: list[str] = Field(
        default_factory=lambda: ["clips_labelled", "raw_recordings"],
        description="Category directory names to scan for source audio files.",
    )
    low_quality_as_negative: bool = Field(
        default=False,
        description="If true, hums below low_quality_threshold are used as negatives.",
    )
    low_quality_threshold: int = Field(
        default=1,
        description="Quality strictly below this value qualifies as low-quality negative.",
    )


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_collection_names: list[str]
    noise_collection_names: list[str]
    split_strategy: str
    seed: int
    min_quality: int
    noise_per_positive: float
    noise_mining: NoiseMiningConfig
    split_fractions: tuple[float, float, float]
    duration_tolerance_s: float = 0.1
    review_gap_s: float = 0.5
    freq_low_hz: int = 0
    freq_high_hz: int = 4000

    @model_validator(mode="after")
    def _validate_split_fractions(self) -> StrategyConfig:
        if not self.target_collection_names:
            raise ValueError("target_collection_names must contain at least one collection")
        if not self.noise_collection_names:
            raise ValueError("noise_collection_names must contain at least one collection")
        if not self.noise_mining.source_category_dirs:
            raise ValueError("noise_mining.source_category_dirs must contain at least one category")

        for collection_name in self.target_collection_names:
            if not collection_name.startswith("audio_collection_"):
                raise ValueError(
                    "target_collection_names entries must start with 'audio_collection_'"
                )
            if not _is_safe_path_segment(collection_name):
                raise ValueError("target_collection_names entries must be safe path segments")

        for collection_name in self.noise_collection_names:
            if not collection_name.startswith("audio_collection_"):
                raise ValueError(
                    "noise_collection_names entries must start with 'audio_collection_'"
                )
            if not _is_safe_path_segment(collection_name):
                raise ValueError("noise_collection_names entries must be safe path segments")

        for category_dir in self.noise_mining.source_category_dirs:
            if not _is_safe_path_segment(category_dir):
                raise ValueError(
                    "noise_mining.source_category_dirs entries must be safe path segments"
                )

        for fraction in self.split_fractions:
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    f"Each split fraction must be in (0, 1), got {self.split_fractions}"
                )
        total = sum(self.split_fractions)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split fractions must sum to ~1.0, got {total:.4f} from {self.split_fractions}"
            )
        return self


class DatasetBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_strategies: list[str]
    strategies: dict[str, StrategyConfig]


def _is_safe_path_segment(value: str) -> bool:
    if not value or value in {".", ".."}:
        return False
    return "/" not in value and "\\" not in value and ".." not in value
