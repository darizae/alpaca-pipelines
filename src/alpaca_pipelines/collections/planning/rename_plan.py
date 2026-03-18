from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.fs import FileSystem
from alpaca_pipelines.collections.parsing.normalizers import normalize_labelled_clip_filename
from alpaca_pipelines.collections.parsing.parse import normalize_segmented_hum_filename
from alpaca_pipelines.collections.paths import (
    LEGACY_CLIPS_DIR_NAMES,
    LEGACY_HUMS_DIR_NAMES,
    CategoryNames,
    find_category_dir,
)
from alpaca_pipelines.recordings import RAW_RECORDINGS_DIR, load_collection_recordings

_RAW_RECORDING_KEY_RE = re.compile(r"^[A-Za-z0-9]+_\d{8}_\d{6}$")


@dataclass(frozen=True)
class RenameOp:
    src: str
    dst: str


@dataclass(frozen=True)
class ClipAuditRow:
    old_path: str
    new_path: str
    subject_token_original: str
    subject_id: str
    date_yyyymmdd: str
    time_hhmmss: str | None
    note: str | None


@dataclass(frozen=True)
class HumAuditRow:
    old_path: str
    new_path: str


@dataclass(frozen=True)
class RawRecordingAuditRow:
    old_path: str
    new_path: str
    recording_key: str
    kind: str


@dataclass(frozen=True)
class RecordingPathUpdate:
    collection: str
    recording_key: str
    wav_path: str
    csv_path: str | None


@dataclass(frozen=True)
class RenamePlan:
    root: str
    ops: list[RenameOp]
    audit: dict[str, Any]


def _detect_plan_collisions(ops: list[RenameOp]) -> None:
    dst_to_src: dict[str, str] = {}
    seen_src: set[str] = set()

    for op in ops:
        if op.src in seen_src:
            raise ValueError(f"Duplicate src in plan: {op.src}")
        seen_src.add(op.src)

        existing = dst_to_src.get(op.dst)
        if existing is not None and existing != op.src:
            raise ValueError(
                f"Destination collision in plan: {op.dst} from {existing} and {op.src}"
            )
        dst_to_src[op.dst] = op.src


def plan_renames_for_collection(
    collection_dir: Path,
    identity_map: IdentityMap,
    category_names: CategoryNames,
    fs: FileSystem,
) -> tuple[
    list[RenameOp],
    list[ClipAuditRow],
    list[HumAuditRow],
    list[RawRecordingAuditRow],
    list[RecordingPathUpdate],
]:
    clips_src = find_category_dir(
        collection_dir, category_names.clips_labelled, LEGACY_CLIPS_DIR_NAMES, fs
    )
    hums_src = find_category_dir(
        collection_dir, category_names.hums_segmented, LEGACY_HUMS_DIR_NAMES, fs
    )

    clip_ops: list[RenameOp] = []
    hum_ops: list[RenameOp] = []
    raw_ops: list[RenameOp] = []
    dir_ops: list[RenameOp] = []

    clip_audit: list[ClipAuditRow] = []
    hum_audit: list[HumAuditRow] = []
    raw_audit: list[RawRecordingAuditRow] = []
    recording_updates: list[RecordingPathUpdate] = []

    if clips_src is not None:
        for clip_path in fs.rglob_wavs(clips_src):
            normalized = normalize_labelled_clip_filename(clip_path, identity_map, fs)
            new_filename = normalized.to_filename()

            if clip_path.name == new_filename:
                continue

            dst_path = clip_path.with_name(new_filename)
            clip_ops.append(RenameOp(src=str(clip_path), dst=str(dst_path)))

            clip_audit.append(
                ClipAuditRow(
                    old_path=str(clip_path.relative_to(collection_dir)),
                    new_path=str(dst_path.relative_to(collection_dir)),
                    subject_token_original=normalized.subject_alias,
                    subject_id=normalized.subject_id,
                    date_yyyymmdd=normalized.date_yyyymmdd,
                    time_hhmmss=normalized.time_hhmmss,
                    note=normalized.note,
                )
            )

    if hums_src is not None:
        for hum_path in fs.rglob_wavs(hums_src):
            canonical_name = normalize_segmented_hum_filename(hum_path)  # validates only
            if hum_path.name == canonical_name:
                continue

            dst_path = hum_path.with_name(canonical_name)
            hum_ops.append(RenameOp(src=str(hum_path), dst=str(dst_path)))
            hum_audit.append(
                HumAuditRow(
                    old_path=str(hum_path.relative_to(collection_dir)),
                    new_path=str(dst_path.relative_to(collection_dir)),
                )
            )

    if clips_src is not None and clips_src.name != category_names.clips_labelled:
        dir_ops.append(
            RenameOp(src=str(clips_src), dst=str(collection_dir / category_names.clips_labelled))
        )

    if hums_src is not None and hums_src.name != category_names.hums_segmented:
        dir_ops.append(
            RenameOp(src=str(hums_src), dst=str(collection_dir / category_names.hums_segmented))
        )

    raw_ops, raw_audit, recording_updates = plan_raw_recording_renames_for_collection(
        collection_dir, fs
    )

    ops = [*clip_ops, *hum_ops, *raw_ops, *dir_ops]
    _detect_plan_collisions(ops)

    return ops, clip_audit, hum_audit, raw_audit, recording_updates


def plan_raw_recording_renames_for_collection(
    collection_dir: Path,
    fs: FileSystem,
) -> tuple[list[RenameOp], list[RawRecordingAuditRow], list[RecordingPathUpdate]]:
    raw_recordings_dir = collection_dir / RAW_RECORDINGS_DIR
    if not (fs.exists(raw_recordings_dir) and fs.is_dir(raw_recordings_dir)):
        return [], [], []

    collection_root = collection_dir.parent
    recordings = load_collection_recordings(collection_dir, fs)
    tracked_files: set[Path] = set()
    for recording in recordings:
        tracked_files.add(
            _resolve_recording_raw_file_path(
                collection_root=collection_root,
                collection_dir=collection_dir,
                raw_path=recording.wav_path,
                field_name="wav_path",
                expected_suffix=".wav",
                fs=fs,
            )
        )
        if recording.csv_path is not None:
            tracked_files.add(
                _resolve_recording_raw_file_path(
                    collection_root=collection_root,
                    collection_dir=collection_dir,
                    raw_path=recording.csv_path,
                    field_name="csv_path",
                    expected_suffix=".csv",
                    fs=fs,
                )
            )

    raw_audio_and_sidecars = sorted(
        entry
        for entry in fs.iterdir(raw_recordings_dir)
        if fs.is_file(entry) and entry.suffix.lower() in {".wav", ".csv"}
    )
    for raw_file in raw_audio_and_sidecars:
        if raw_file not in tracked_files:
            raise ValueError(
                "Raw file is not represented in recordings.json: "
                f"{raw_file} (collection={collection_dir.name})"
            )

    ops: list[RenameOp] = []
    audit_rows: list[RawRecordingAuditRow] = []
    updates: list[RecordingPathUpdate] = []
    seen_canonical_stems: set[str] = set()

    for recording in recordings:
        canonical_stem = _canonical_raw_stem_from_key(recording.key)
        if canonical_stem in seen_canonical_stems:
            raise ValueError(
                f"Duplicate recording key canonical stem in {collection_dir.name}: {canonical_stem}"
            )
        seen_canonical_stems.add(canonical_stem)

        wav_src = _resolve_recording_raw_file_path(
            collection_root=collection_root,
            collection_dir=collection_dir,
            raw_path=recording.wav_path,
            field_name="wav_path",
            expected_suffix=".wav",
            fs=fs,
        )
        wav_dst = raw_recordings_dir / f"{canonical_stem}.WAV"
        csv_src: Path | None = None
        csv_dst: Path | None = None

        if recording.csv_path is not None:
            csv_src = _resolve_recording_raw_file_path(
                collection_root=collection_root,
                collection_dir=collection_dir,
                raw_path=recording.csv_path,
                field_name="csv_path",
                expected_suffix=".csv",
                fs=fs,
            )
            csv_dst = raw_recordings_dir / f"{canonical_stem}.CSV"

        changed = False
        if wav_src != wav_dst:
            ops.append(RenameOp(src=str(wav_src), dst=str(wav_dst)))
            audit_rows.append(
                RawRecordingAuditRow(
                    old_path=str(wav_src.relative_to(collection_dir)),
                    new_path=str(wav_dst.relative_to(collection_dir)),
                    recording_key=recording.key,
                    kind="wav",
                )
            )
            changed = True

        if csv_src is not None and csv_dst is not None and csv_src != csv_dst:
            ops.append(RenameOp(src=str(csv_src), dst=str(csv_dst)))
            audit_rows.append(
                RawRecordingAuditRow(
                    old_path=str(csv_src.relative_to(collection_dir)),
                    new_path=str(csv_dst.relative_to(collection_dir)),
                    recording_key=recording.key,
                    kind="csv",
                )
            )
            changed = True

        if changed:
            updates.append(
                RecordingPathUpdate(
                    collection=collection_dir.name,
                    recording_key=recording.key,
                    wav_path=_root_relative_path(collection_root, wav_dst),
                    csv_path=(
                        _root_relative_path(collection_root, csv_dst)
                        if csv_dst is not None
                        else None
                    ),
                )
            )

    return ops, audit_rows, updates


def _resolve_recording_raw_file_path(
    *,
    collection_root: Path,
    collection_dir: Path,
    raw_path: str,
    field_name: str,
    expected_suffix: str,
    fs: FileSystem,
) -> Path:
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ValueError(
            f"{collection_dir.name}: recordings.json {field_name} must be root-relative: {raw_path}"
        )
    absolute_path = collection_root / relative_path
    raw_recordings_dir = collection_dir / RAW_RECORDINGS_DIR
    if not absolute_path.is_relative_to(raw_recordings_dir):
        raise ValueError(
            f"{collection_dir.name}: recordings.json {field_name} must point under "
            f"{RAW_RECORDINGS_DIR}: {raw_path}"
        )
    if absolute_path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{collection_dir.name}: recordings.json {field_name} has unexpected extension: "
            f"{raw_path}"
        )
    if not fs.exists(absolute_path):
        raise FileNotFoundError(
            f"{collection_dir.name}: recordings.json {field_name} points to missing file: "
            f"{absolute_path}"
        )
    return absolute_path


def _canonical_raw_stem_from_key(recording_key: str) -> str:
    if _RAW_RECORDING_KEY_RE.match(recording_key) is None:
        raise ValueError(
            "Unsupported recording key for raw canonicalization "
            f"(expected <subject_id>_<YYYYMMDD>_<HHMMSS>): {recording_key}"
        )
    return recording_key


def _root_relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root))
