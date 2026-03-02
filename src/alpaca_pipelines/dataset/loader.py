"""
Dataset loading from the folder-based persistence layer.

Reads manifest.json and splits/*.csv to build dataset handles
that the training and evaluation executors consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpaca_pipelines.contracts import (
    MANIFEST_FILENAME,
    SNIPPETS_DIR,
    SPLITS_DIR,
    DatasetManifest,
)
from alpaca_pipelines.io_utils import read_csv_column, read_json
from alpaca_pipelines.paths import validate_basename


@dataclass(frozen=True)
class SplitFiles:
    """File lists for each data split."""

    train: list[str]
    val: list[str]
    test: list[str]

    @property
    def all_splits(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class DatasetHandle:
    """Validated reference to a dataset on disk.

    This is the primary interface between the persistence layer and
    the pipeline executors.  It provides validated paths and file
    lists without loading any audio data.
    """

    dataset_dir: Path
    manifest: DatasetManifest
    splits: SplitFiles
    snippets_dir: Path
    classes: list[str]
    class_to_index: dict[str, int]

    @property
    def strategy_name(self) -> str:
        return self.manifest.meta.strategy_name

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def split_file_count(self, split_name: str) -> int:
        return len(self.splits.all_splits[split_name])


def _extract_classes_from_manifest(manifest: DatasetManifest) -> tuple[list[str], dict[str, int]]:
    """Derive class list and mapping from manifest snippets."""
    raw_classifications = sorted({snippet.classification for snippet in manifest.snippets})
    classifications: list[str] = list(map(str, raw_classifications))
    class_to_index: dict[str, int] = {}
    if len(classifications) == 2 and "target" in classifications and "noise" in classifications:
        class_to_index = {"noise": 0, "target": 1}
    else:
        for index, class_name in enumerate(classifications):
            class_to_index[class_name] = index
    return classifications, class_to_index


def _validate_manifest_snippets(
    manifest: DatasetManifest,
    snippets_dir: Path,
) -> dict[str, str]:
    """Validate all manifest snippet entries.

    Returns a mapping of filename → split (if assigned) for
    cross-referencing with split CSV files.
    """
    manifest_filenames: dict[str, str] = {}
    for snippet in manifest.snippets:
        validate_basename(snippet.filename)
        snippet_path = snippets_dir / snippet.filename
        if not snippet_path.is_file():
            raise FileNotFoundError(
                "Snippet file listed in manifest does not exist: {}".format(snippet_path)
            )
        split_value = snippet.split if snippet.split is not None else ""
        manifest_filenames[snippet.filename] = split_value

    if manifest.meta.n_snippets != len(manifest.snippets):
        raise ValueError(
            "Manifest meta.n_snippets ({}) does not match actual snippet count ({})".format(
                manifest.meta.n_snippets, len(manifest.snippets)
            )
        )

    return manifest_filenames


def _load_split_file(
    dataset_dir: Path,
    split_name: str,
    snippets_dir: Path,
    manifest_filenames: dict[str, str],
) -> list[str]:
    """Load and validate a split CSV file.

    Checks that each filename is a safe basename, exists as a wav
    file under snippets_dir, and appears in the manifest.
    """
    split_path = dataset_dir / SPLITS_DIR / "{}.csv".format(split_name)
    if not split_path.is_file():
        raise FileNotFoundError("Split file not found: {}".format(split_path))
    filenames = read_csv_column(split_path)
    for filename in filenames:
        validate_basename(filename)
        if filename not in manifest_filenames:
            raise ValueError(
                "Split file '{}' references '{}' which is not in the manifest".format(
                    split_path, filename
                )
            )
        snippet_path = snippets_dir / filename
        if not snippet_path.is_file():
            raise FileNotFoundError(
                "Snippet file listed in {} split does not exist: {}".format(
                    split_name, snippet_path
                )
            )
    return filenames


def load_dataset_handle(dataset_dir: Path) -> DatasetHandle:
    """Load a complete dataset handle from disk.

    Validates manifest, splits, snippet existence, and cross-references
    split files against manifest entries.  Does NOT load any audio data.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(dataset_dir))

    manifest_path = dataset_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError("Manifest not found: {}".format(manifest_path))

    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, dict):
        raise ValueError("Expected JSON object in manifest: {}".format(manifest_path))
    manifest = DatasetManifest.model_validate(raw_manifest)

    snippets_dir = dataset_dir / SNIPPETS_DIR
    if not snippets_dir.is_dir():
        raise FileNotFoundError("Snippets directory not found: {}".format(snippets_dir))

    manifest_filenames = _validate_manifest_snippets(manifest, snippets_dir)

    train_files = _load_split_file(dataset_dir, "train", snippets_dir, manifest_filenames)
    val_files = _load_split_file(dataset_dir, "val", snippets_dir, manifest_filenames)
    test_files = _load_split_file(dataset_dir, "test", snippets_dir, manifest_filenames)

    classes, class_to_index = _extract_classes_from_manifest(manifest)

    return DatasetHandle(
        dataset_dir=dataset_dir,
        manifest=manifest,
        splits=SplitFiles(train=train_files, val=val_files, test=test_files),
        snippets_dir=snippets_dir,
        classes=classes,
        class_to_index=class_to_index,
    )
