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
from alpaca_pipelines.recordings import (
    SourceRecording,
    compute_recording_counts,
    derive_source_recording_key,
    load_collection_recordings,
)


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
    source_recording_key: str | None


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
    collection_recordings = load_collection_recordings(collection_dir, fs)
    recordings_by_key = {recording.key: recording for recording in collection_recordings}

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
        source_recording_key = None
        if recording_time is not None:
            candidate_key = derive_source_recording_key(
                subject_id=subject_id,
                recording_date=recording_date,
                recording_time=recording_time,
            )
            if candidate_key in recordings_by_key:
                source_recording_key = candidate_key

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
                source_recording_key=source_recording_key,
            )
        )
        hum_uid += 1

    n_recordings, n_recordings_with_sidecar = compute_recording_counts(collection_recordings)
    meta: dict[str, Any] = {
        "collection": collection_dir.name,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_hums": len(entries),
        "n_recordings": n_recordings,
        "n_recordings_with_sidecar": n_recordings_with_sidecar,
        "min_source_quality_to_keep": config.min_source_quality_to_keep,
        "persistence_root_contract": "hum_path is relative to ALPACA_COLLECTION_ROOT",
    }

    return {
        "meta": meta,
        "entries": [e.__dict__ for e in entries],
        "recordings": [recording.model_dump() for recording in collection_recordings],
    }


def merge_indexes(indexes: list[dict[str, Any]]) -> dict[str, Any]:
    merged_entries: list[dict[str, Any]] = []
    merged_recordings: dict[str, SourceRecording] = {}
    for index in indexes:
        merged_entries.extend(index["entries"])
        for raw_recording in index.get("recordings", []):
            recording = SourceRecording.model_validate(raw_recording)
            merged_recordings[recording.key] = recording

    n_recordings, n_recordings_with_sidecar = compute_recording_counts(
        list(merged_recordings.values())
    )
    meta: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_collections": len(indexes),
        "n_total_hums": len(merged_entries),
        "n_recordings": n_recordings,
        "n_recordings_with_sidecar": n_recordings_with_sidecar,
    }
    return {
        "meta": meta,
        "entries": merged_entries,
        "recordings": [recording.model_dump() for recording in merged_recordings.values()],
    }
