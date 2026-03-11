from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpaca_pipelines.collections.fs import FileSystem


@dataclass(frozen=True)
class CategoryNames:
    clips_labelled: str = "clips_labelled"
    hums_segmented: str = "hums_segmented"


LEGACY_CLIPS_DIR_NAMES: tuple[str, ...] = ("labelled_recordings",)
LEGACY_HUMS_DIR_NAMES: tuple[str, ...] = ("segmented wav_onlyhums", "segmented_wav_onlyhums")


def find_collection_dirs(root: Path, fs: FileSystem) -> list[Path]:
    if not fs.exists(root):
        raise FileNotFoundError(root)
    collections = [
        p for p in fs.iterdir(root) if fs.is_dir(p) and p.name.startswith("audio_collection_")
    ]
    if not collections:
        raise FileNotFoundError(f"No audio_collection_* directories found under {root}")
    return sorted(collections)


def find_category_dir(
    collection_dir: Path,
    desired: str,
    legacy_names: tuple[str, ...],
    fs: FileSystem,
) -> Path | None:
    desired_path = collection_dir / desired
    if fs.exists(desired_path):
        return desired_path
    for legacy in legacy_names:
        legacy_path = collection_dir / legacy
        if fs.exists(legacy_path):
            return legacy_path
    return None
