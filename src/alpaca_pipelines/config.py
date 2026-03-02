"""
Pipeline environment configuration.

Validates that all required paths and environment variables are present
and that the persistence layer is accessible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_ENV_VARS: tuple[str, ...] = (
    "ALPACA_COLLECTION_ROOT",
    "ALPACA_MERGED_INDEX",
    "ALPACA_DATASETS_ROOT",
    "ALPACA_RUNS_ROOT",
)


@dataclass(frozen=True)
class PipelineEnvironment:
    """Validated pipeline environment with all required paths."""

    collection_root: Path
    merged_index_path: Path
    datasets_root: Path
    runs_root: Path

    @classmethod
    def from_env(cls) -> PipelineEnvironment:
        """Build from environment variables, failing if any are missing."""
        missing = [var for var in _REQUIRED_ENV_VARS if not os.environ.get(var)]
        if missing:
            raise EnvironmentError(
                "Missing required environment variables: {}".format(", ".join(missing))
            )
        return cls(
            collection_root=Path(os.environ["ALPACA_COLLECTION_ROOT"]),
            merged_index_path=Path(os.environ["ALPACA_MERGED_INDEX"]),
            datasets_root=Path(os.environ["ALPACA_DATASETS_ROOT"]),
            runs_root=Path(os.environ["ALPACA_RUNS_ROOT"]),
        )

    @classmethod
    def from_explicit(
        cls,
        collection_root: str | Path,
        merged_index_path: str | Path,
        datasets_root: str | Path,
        runs_root: str | Path,
    ) -> PipelineEnvironment:
        """Build from explicit parameters (for API usage)."""
        return cls(
            collection_root=Path(collection_root),
            merged_index_path=Path(merged_index_path),
            datasets_root=Path(datasets_root),
            runs_root=Path(runs_root),
        )

    def validate(self) -> None:
        """Validate that all required paths exist and are accessible."""
        if not self.collection_root.is_dir():
            raise FileNotFoundError(
                "ALPACA_COLLECTION_ROOT does not exist: {}".format(self.collection_root)
            )
        if not self.merged_index_path.is_file():
            raise FileNotFoundError(
                "ALPACA_MERGED_INDEX does not exist: {}".format(self.merged_index_path)
            )

        collection_root_resolved = self.collection_root.resolve()
        merged_index_resolved = self.merged_index_path.resolve()
        try:
            merged_index_resolved.relative_to(collection_root_resolved)
        except ValueError:
            raise ValueError(
                "ALPACA_MERGED_INDEX must be under ALPACA_COLLECTION_ROOT: {} "
                "is not under {}".format(merged_index_resolved, collection_root_resolved)
            ) from None

        if not self.datasets_root.is_dir():
            raise FileNotFoundError(
                "ALPACA_DATASETS_ROOT does not exist: {}".format(self.datasets_root)
            )
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def resolve_dataset_dir(self, dataset_name: str) -> Path:
        """Resolve a dataset directory under ALPACA_DATASETS_ROOT."""
        dataset_dir = self.datasets_root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError("Dataset directory does not exist: {}".format(dataset_dir))
        return dataset_dir
