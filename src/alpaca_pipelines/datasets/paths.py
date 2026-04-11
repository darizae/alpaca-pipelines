from __future__ import annotations

from pathlib import Path, PurePosixPath

from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem

SNIPPETS_DIR: str = "snippets"
SPLITS_DIR: str = "splits"
REVIEW_DIR: str = "review"
MANIFEST_FILENAME: str = "manifest.json"
REVIEW_TARGET_CONCAT_FILENAME: str = "review_target_concat.wav"
REVIEW_NOISE_CONCAT_FILENAME: str = "review_noise_concat.wav"
REVIEW_TARGET_SELECTION_TABLE_FILENAME: str = "review_target_selection_table.txt"
REVIEW_NOISE_SELECTION_TABLE_FILENAME: str = "review_noise_selection_table.txt"
CORRECTIONS_APPLIED_FILENAME: str = "corrections_applied.json"

HUMS_SEGMENTED_DIR: str = "hums_segmented"


def validate_relative_path(relative_path: str, root: Path) -> Path:
    posix_relative = PurePosixPath(relative_path)
    if posix_relative.is_absolute():
        raise ValueError(f"Absolute path not allowed: {relative_path}")
    if ".." in posix_relative.parts:
        raise ValueError(f"Path traversal not allowed: {relative_path}")

    root_posix = PurePosixPath(root.as_posix())
    joined = root_posix / posix_relative

    try:
        joined.relative_to(root_posix)
    except ValueError:
        raise ValueError(
            f"Path escapes root: {relative_path} resolves to {joined}, root is {root_posix}"
        ) from None

    return Path(str(joined))


def find_collection_dirs(root: Path, fs: FileSystem = _DEFAULT_FS) -> list[Path]:
    if not fs.exists(root):
        raise FileNotFoundError(root)

    collections = [
        p for p in fs.iterdir(root) if fs.is_dir(p) and p.name.startswith("audio_collection_")
    ]
    if not collections:
        raise FileNotFoundError(f"No audio_collection_* directories found under {root}")
    return sorted(collections)


def rglob_wavs(root: Path, fs: FileSystem = _DEFAULT_FS) -> list[Path]:
    return fs.rglob_wavs(root)


def ensure_dataset_dirs(dataset_dir: Path, fs: FileSystem = _DEFAULT_FS) -> None:
    fs.makedirs(dataset_dir / SNIPPETS_DIR)
    fs.makedirs(dataset_dir / SPLITS_DIR)
    fs.makedirs(dataset_dir / REVIEW_DIR)
