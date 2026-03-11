from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpaca_pipelines.collections.fs import FileSystem
from alpaca_pipelines.collections.paths import (
    LEGACY_CLIPS_DIR_NAMES,
    LEGACY_HUMS_DIR_NAMES,
    CategoryNames,
    find_category_dir,
)


@dataclass(frozen=True)
class CollectionScan:
    collection_name: str
    collection_dir: Path
    clips_dir: Path
    hums_dir: Path
    clip_files: list[Path]
    hum_files: list[Path]


def scan_collection(
    collection_dir: Path,
    category_names: CategoryNames,
    fs: FileSystem,
) -> CollectionScan:
    clips_dir = find_category_dir(
        collection_dir, category_names.clips_labelled, LEGACY_CLIPS_DIR_NAMES, fs
    )
    if clips_dir is None:
        raise FileNotFoundError(f"Missing clips directory under {collection_dir}")

    hums_dir = find_category_dir(
        collection_dir, category_names.hums_segmented, LEGACY_HUMS_DIR_NAMES, fs
    )
    if hums_dir is None:
        raise FileNotFoundError(f"Missing hums directory under {collection_dir}")

    return CollectionScan(
        collection_name=collection_dir.name,
        collection_dir=collection_dir,
        clips_dir=clips_dir,
        hums_dir=hums_dir,
        clip_files=fs.rglob_wavs(clips_dir),
        hum_files=fs.rglob_wavs(hums_dir),
    )
