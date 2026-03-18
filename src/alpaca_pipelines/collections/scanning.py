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
from alpaca_pipelines.recordings import RAW_RECORDINGS_DIR


@dataclass(frozen=True)
class CollectionScan:
    collection_name: str
    collection_dir: Path
    clips_dir: Path | None
    hums_dir: Path | None
    raw_recordings_dir: Path | None
    clip_files: list[Path]
    hum_files: list[Path]
    raw_recording_files: list[Path]
    has_clips: bool
    has_hums: bool
    has_raw_recordings: bool
    status: str


def scan_collection(
    collection_dir: Path,
    category_names: CategoryNames,
    fs: FileSystem,
) -> CollectionScan:
    clips_dir = find_category_dir(
        collection_dir, category_names.clips_labelled, LEGACY_CLIPS_DIR_NAMES, fs
    )
    hums_dir = find_category_dir(
        collection_dir, category_names.hums_segmented, LEGACY_HUMS_DIR_NAMES, fs
    )
    raw_recordings_dir = collection_dir / RAW_RECORDINGS_DIR
    has_raw_recordings = fs.exists(raw_recordings_dir) and fs.is_dir(raw_recordings_dir)
    has_clips = clips_dir is not None
    has_hums = hums_dir is not None
    clip_files = fs.rglob_wavs(clips_dir) if clips_dir is not None else []
    hum_files = fs.rglob_wavs(hums_dir) if hums_dir is not None else []
    raw_recording_files = fs.rglob_wavs(raw_recordings_dir) if has_raw_recordings else []

    if has_clips and has_hums:
        status = "ready"
    elif has_clips:
        status = "clips_only"
    elif has_hums:
        status = "hums_only"
    elif has_raw_recordings:
        status = "raw_only"
    else:
        status = "empty"

    return CollectionScan(
        collection_name=collection_dir.name,
        collection_dir=collection_dir,
        clips_dir=clips_dir,
        hums_dir=hums_dir,
        raw_recordings_dir=raw_recordings_dir if has_raw_recordings else None,
        clip_files=clip_files,
        hum_files=hum_files,
        raw_recording_files=raw_recording_files,
        has_clips=has_clips,
        has_hums=has_hums,
        has_raw_recordings=has_raw_recordings,
        status=status,
    )
