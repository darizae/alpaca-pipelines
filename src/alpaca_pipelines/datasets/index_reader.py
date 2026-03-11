from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.io_utils import read_json


@dataclass(frozen=True)
class HumIndexEntry:
    collection: str
    subject_id: str
    recording_date: str
    recording_time: str | None
    hum_path: str
    hum_start_s: float
    hum_end_s: float
    source_quality: int
    keep: bool
    hum_uid: int


@dataclass(frozen=True)
class IndexPayload:
    entries: list[HumIndexEntry]
    n_collections: int


def _parse_entry(raw: dict[str, Any]) -> HumIndexEntry:
    return HumIndexEntry(
        collection=str(raw["collection"]),
        subject_id=str(raw["subject_id"]),
        recording_date=str(raw["recording_date"]),
        recording_time=str(raw["recording_time"]) if raw.get("recording_time") else None,
        hum_path=str(raw["hum_path"]),
        hum_start_s=float(raw["hum_start_s"]),
        hum_end_s=float(raw["hum_end_s"]),
        source_quality=int(raw["source_quality"]),
        keep=bool(raw["keep"]),
        hum_uid=int(raw["hum_uid"]),
    )


def load_merged_index(path: Path, fs: FileSystem = _DEFAULT_FS) -> IndexPayload:
    data = read_json(path, fs)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at top level: {path}")

    meta = data.get("meta", {})
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError(f"Expected 'entries' to be a list: {path}")

    entries = [_parse_entry(e) for e in raw_entries]
    n_collections = int(meta.get("n_collections", 0))

    return IndexPayload(entries=entries, n_collections=n_collections)
