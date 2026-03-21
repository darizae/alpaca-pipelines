from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.collections.parsing.normalizers import wav_duration_seconds
from alpaca_pipelines.recordings import (
    COLLECTION_RECORDINGS_FILENAME,
    RAW_RECORDINGS_DIR,
    SourceRecording,
    derive_source_recording_key_from_stem,
    parse_audiomoth_csv,
    parse_device_txt,
    parse_settings_txt,
    recording_path,
    stem_to_iso_timestamp,
    write_collection_recordings,
)


@dataclass(frozen=True)
class RawImportResult:
    imported_batches: list[str]
    imported_collection_dirs: list[str]
    imported_recordings: list[SourceRecording]
    matched_csv_count: int
    missing_csv_count: int


def find_raw_batch_dirs(root: Path, fs: FileSystem = _DEFAULT_FS) -> list[Path]:
    if not fs.exists(root):
        raise FileNotFoundError(root)
    raw_batches: list[Path] = []
    for entry in sorted(fs.iterdir(root)):
        if not fs.is_dir(entry):
            continue
        if entry.name.startswith("audio_collection_"):
            continue
        if any(path.parent == entry for path in fs.rglob_wavs(entry)):
            raw_batches.append(entry)
    return raw_batches


def import_raw_batches(
    root: Path,
    identity_map: IdentityMap,
    fs: FileSystem = _DEFAULT_FS,
) -> RawImportResult:
    imported_batches: list[str] = []
    imported_collection_dirs: list[str] = []
    imported_recordings: list[SourceRecording] = []
    matched_csv_count = 0
    missing_csv_count = 0

    for batch_dir in find_raw_batch_dirs(root, fs):
        subject_alias, deployment_token = _parse_batch_dir_name(batch_dir.name)
        subject_id = identity_map.normalize_subject(subject_alias)
        collection_dir = root / f"audio_collection_{batch_dir.name}"
        raw_recordings_dir = collection_dir / RAW_RECORDINGS_DIR

        fs.makedirs(collection_dir)
        fs.makedirs(raw_recordings_dir)

        device_src = _find_optional(batch_dir, "DEVICE.TXT", fs)
        settings_src = _find_optional(batch_dir, "SETTINGS.txt", fs)
        log_src = _find_optional(batch_dir, "LOG.TXT", fs)

        device_dst = _copy_optional(device_src, raw_recordings_dir, fs)
        settings_dst = _copy_optional(settings_src, raw_recordings_dir, fs)
        log_dst = _copy_optional(log_src, raw_recordings_dir, fs)

        device_info = parse_device_txt(device_dst, fs) if device_dst is not None else {}
        settings = parse_settings_txt(settings_dst, fs) if settings_dst is not None else None

        batch_recordings: list[SourceRecording] = []
        for wav_src in sorted(path for path in fs.iterdir(batch_dir) if _is_wav_file(path, fs)):
            wav_dst = raw_recordings_dir / wav_src.name
            _copy_file_idempotent(wav_src, wav_dst, fs)

            stem = wav_src.stem
            key = derive_source_recording_key_from_stem(subject_id, stem)
            csv_src = _find_matching_sidecar(batch_dir, stem, ".csv", fs)
            csv_dst = None
            track_points = None
            if csv_src is not None:
                csv_dst = raw_recordings_dir / csv_src.name
                _copy_file_idempotent(csv_src, csv_dst, fs)
                track_points = parse_audiomoth_csv(csv_dst, fs)
                matched_csv_count += 1
            else:
                missing_csv_count += 1

            duration_seconds = round(wav_duration_seconds(wav_dst, fs), 4)
            sample_rate = _sample_rate_from_settings(settings)
            total_samples = (
                track_points[-1].total_samples
                if track_points
                else (
                    int(round(duration_seconds * sample_rate)) if sample_rate is not None else None
                )
            )
            start_time = (
                track_points[0].audiomoth_time if track_points else stem_to_iso_timestamp(stem)
            )
            end_time = (
                track_points[-1].audiomoth_time
                if track_points
                else _end_time_from_start(start_time, duration_seconds)
            )

            recording = SourceRecording(
                key=key,
                collection=collection_dir.name,
                subject_id=subject_id,
                deployment_token=deployment_token,
                wav_path=recording_path(root, wav_dst),
                csv_path=recording_path(root, csv_dst) if csv_dst is not None else None,
                device_path=recording_path(root, device_dst) if device_dst is not None else None,
                settings_path=(
                    recording_path(root, settings_dst) if settings_dst is not None else None
                ),
                log_path=recording_path(root, log_dst) if log_dst is not None else None,
                device_id=device_info.get("device_id"),
                firmware_version=device_info.get("firmware_version"),
                firmware_description=device_info.get("firmware_description"),
                settings=settings,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                total_samples=total_samples,
                track_points=track_points,
            )
            batch_recordings.append(recording)

        _write_recordings_idempotent(collection_dir, batch_recordings, fs)
        imported_batches.append(batch_dir.name)
        imported_collection_dirs.append(str(collection_dir))
        imported_recordings.extend(batch_recordings)

    return RawImportResult(
        imported_batches=imported_batches,
        imported_collection_dirs=imported_collection_dirs,
        imported_recordings=imported_recordings,
        matched_csv_count=matched_csv_count,
        missing_csv_count=missing_csv_count,
    )


def _parse_batch_dir_name(name: str) -> tuple[str, str | None]:
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Raw batch directory must include subject alias and deployment token: {name}"
        )
    return parts[0], parts[1] or None


def _is_wav_file(path: Path, fs: FileSystem) -> bool:
    return fs.is_file(path) and path.suffix.lower() == ".wav"


def _find_optional(directory: Path, filename: str, fs: FileSystem) -> Path | None:
    for entry in fs.iterdir(directory):
        if fs.is_file(entry) and entry.name.lower() == filename.lower():
            return entry
    return None


def _find_matching_sidecar(
    directory: Path,
    stem: str,
    suffix: str,
    fs: FileSystem,
) -> Path | None:
    desired = f"{stem}{suffix}"
    for entry in fs.iterdir(directory):
        if fs.is_file(entry) and entry.name.lower() == desired.lower():
            return entry
    return None


def _copy_optional(src: Path | None, dst_dir: Path, fs: FileSystem) -> Path | None:
    if src is None:
        return None
    dst = dst_dir / src.name
    _copy_file_idempotent(src, dst, fs)
    return dst


def _copy_file_idempotent(src: Path, dst: Path, fs: FileSystem) -> None:
    if fs.exists(dst):
        if _file_bytes(src, fs) != _file_bytes(dst, fs):
            raise FileExistsError(f"Target exists with different content: {dst}")
        return
    reader = fs.open_read(src)
    try:
        writer = fs.open_write(dst)
        try:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)
        finally:
            writer.close()
    finally:
        reader.close()


def _file_bytes(path: Path, fs: FileSystem) -> bytes:
    handle = fs.open_read(path)
    try:
        return handle.read()
    finally:
        handle.close()


def _sample_rate_from_settings(settings: dict[str, object] | None) -> int | None:
    if settings is None:
        return None
    raw_value = settings.get("sampleRate")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        return None
    return int(raw_value)


def _end_time_from_start(start_time: str, duration_seconds: float) -> str:
    started_at = datetime.fromisoformat(start_time)
    return (started_at + timedelta(seconds=duration_seconds)).isoformat(timespec="milliseconds")


def _write_recordings_idempotent(
    collection_dir: Path,
    recordings: list[SourceRecording],
    fs: FileSystem,
) -> None:
    path = collection_dir / COLLECTION_RECORDINGS_FILENAME
    payload = {"recordings": [recording.model_dump() for recording in recordings]}
    if fs.exists(path):
        existing_content = fs.read_text(path)
        existing_payload = json.loads(existing_content)
        if existing_payload != payload:
            raise FileExistsError(f"Recording metadata differs from existing file: {path}")
        return
    write_collection_recordings(collection_dir, recordings, fs)
