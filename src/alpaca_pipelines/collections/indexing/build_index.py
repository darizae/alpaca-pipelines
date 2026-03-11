from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from alpaca_pipelines.collections.config import StandardizerConfig
from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.fs import FileSystem
from alpaca_pipelines.collections.parsing.parse import parse_canonical_hum_filename
from alpaca_pipelines.collections.parsing.patterns import CANONICAL_CLIP_RE


@dataclass(frozen=True)
class IndexEntry:
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


def build_collection_index(
    *,
    persistence_root: Path,
    collection_dir: Path,
    hums_dir: Path,
    identity_map: IdentityMap,
    config: StandardizerConfig,
    fs: FileSystem,
) -> dict[str, Any]:
    """
    Build an index for one collection.

    IMPORTANT: hum_path is written relative to the persistence root (ALPACA_COLLECTION_ROOT),
    so that downstream tools can resolve: persistence_root / hum_path
    """
    if collection_dir.parent != persistence_root:
        raise ValueError(
            "Collection directory is not directly under persistence root. "
            f"persistence_root={persistence_root} collection_dir={collection_dir}"
        )

    hum_files = fs.rglob_wavs(hums_dir)

    entries: list[IndexEntry] = []
    hum_uid = 1

    for hum_path in hum_files:
        parsed = parse_canonical_hum_filename(hum_path.name)

        clip_match = CANONICAL_CLIP_RE.match(parsed.clip_filename)
        if clip_match is None:
            raise ValueError(
                f"Hum embeds non-canonical clip filename: {hum_path} -> {parsed.clip_filename}"
            )

        subject_token = clip_match["subject"]
        subject_id = identity_map.normalize_subject(subject_token)

        date_yyyymmdd = clip_match["date"]
        time_hhmmss = clip_match.group("time")

        recording_date = f"{date_yyyymmdd[0:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"
        recording_time = (
            f"{time_hhmmss[0:2]}:{time_hhmmss[2:4]}:{time_hhmmss[4:6]}" if time_hhmmss else None
        )

        keep = True
        if config.min_source_quality_to_keep is not None:
            keep = parsed.source_quality >= config.min_source_quality_to_keep

        # Root-relative path (includes audio_collection_*/ prefix)
        hum_rel = hum_path.relative_to(persistence_root)

        entries.append(
            IndexEntry(
                collection=collection_dir.name,
                subject_id=subject_id,
                recording_date=recording_date,
                recording_time=recording_time,
                hum_path=str(hum_rel),
                hum_start_s=parsed.hum_start_s,
                hum_end_s=parsed.hum_end_s,
                source_quality=parsed.source_quality,
                keep=keep,
                hum_uid=hum_uid,
            )
        )
        hum_uid += 1

    meta: dict[str, Any] = {
        "collection": collection_dir.name,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_hums": len(entries),
        "min_source_quality_to_keep": config.min_source_quality_to_keep,
        "persistence_root_contract": "hum_path is relative to ALPACA_COLLECTION_ROOT",
    }

    return {"meta": meta, "entries": [e.__dict__ for e in entries]}


def merge_indexes(indexes: list[dict[str, Any]]) -> dict[str, Any]:
    merged_entries: list[dict[str, Any]] = []
    for index in indexes:
        merged_entries.extend(index["entries"])

    meta: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_collections": len(indexes),
        "n_total_hums": len(merged_entries),
    }
    return {"meta": meta, "entries": merged_entries}
