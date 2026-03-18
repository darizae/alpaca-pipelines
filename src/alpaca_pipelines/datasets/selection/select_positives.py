from __future__ import annotations

from itertools import count
from pathlib import Path

from alpaca_pipelines.datasets.audio_utils import validate_snippet_duration
from alpaca_pipelines.datasets.contracts import SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.index_reader import HumIndexEntry
from alpaca_pipelines.datasets.paths import validate_relative_path
from alpaca_pipelines.datasets.recording_metadata import with_recording_window
from alpaca_pipelines.recordings import SourceRecording


def _session_key(entry: HumIndexEntry) -> str:
    date_compact = entry.recording_date.replace("-", "")
    return f"{entry.subject_id}_{date_compact}"


def _snippet_filename(classification: str, label: str, uid: int, collection: str) -> str:
    return f"{classification}-{label}_{uid:06d}_{collection}.wav"


def select_positives(
    entries: list[HumIndexEntry],
    recordings_by_key: dict[str, SourceRecording],
    min_quality: int,
    collection_root: Path,
    snippets_dir: Path,
    uid_counter: count[int],
    duration_tolerance_s: float,
    fs: FileSystem = _DEFAULT_FS,
) -> list[SnippetEntry]:
    selected: list[SnippetEntry] = []

    for entry in entries:
        if not entry.keep:
            continue
        if entry.source_quality < min_quality:
            continue

        hum_wav_path = validate_relative_path(entry.hum_path, collection_root)
        if not fs.exists(hum_wav_path):
            raise FileNotFoundError(f"Hum wav not found: {hum_wav_path}")

        actual_duration = validate_snippet_duration(
            hum_wav_path,
            entry.hum_start_s,
            entry.hum_end_s,
            duration_tolerance_s,
            fs,
        )

        uid = next(uid_counter)
        label = f"Q{entry.source_quality}"
        filename = _snippet_filename("target", label, uid, entry.collection)

        destination = snippets_dir / filename
        _copy_wav(hum_wav_path, destination, fs)

        snippet = SnippetEntry(
            uid=uid,
            filename=filename,
            classification="target",
            source_type="hum",
            source_path=entry.hum_path,
            start_s=0.0,
            end_s=actual_duration,
            duration_s=actual_duration,
            quality=entry.source_quality,
            subject_id=entry.subject_id,
            recording_date=entry.recording_date,
            recording_time=entry.recording_time,
            collection=entry.collection,
            session_key=_session_key(entry),
        )
        selected.append(
            with_recording_window(
                snippet,
                recordings_by_key.get(entry.source_recording_key)
                if entry.source_recording_key
                else None,
                start_offset_s=entry.hum_start_s,
                end_offset_s=entry.hum_end_s,
            )
        )

    return selected


def select_low_quality_as_negatives(
    entries: list[HumIndexEntry],
    recordings_by_key: dict[str, SourceRecording],
    low_quality_threshold: int,
    collection_root: Path,
    snippets_dir: Path,
    uid_counter: count[int],
    duration_tolerance_s: float,
    fs: FileSystem = _DEFAULT_FS,
) -> list[SnippetEntry]:
    selected: list[SnippetEntry] = []

    for entry in entries:
        if not entry.keep:
            continue
        if entry.source_quality >= low_quality_threshold:
            continue

        hum_wav_path = validate_relative_path(entry.hum_path, collection_root)
        if not fs.exists(hum_wav_path):
            raise FileNotFoundError(f"Hum wav not found: {hum_wav_path}")

        actual_duration = validate_snippet_duration(
            hum_wav_path,
            entry.hum_start_s,
            entry.hum_end_s,
            duration_tolerance_s,
            fs,
        )

        uid = next(uid_counter)
        filename = _snippet_filename("noise", "lowq", uid, entry.collection)

        destination = snippets_dir / filename
        _copy_wav(hum_wav_path, destination, fs)

        snippet = SnippetEntry(
            uid=uid,
            filename=filename,
            classification="noise",
            source_type="low_quality_hum",
            source_path=entry.hum_path,
            start_s=0.0,
            end_s=actual_duration,
            duration_s=actual_duration,
            quality=entry.source_quality,
            subject_id=entry.subject_id,
            recording_date=entry.recording_date,
            recording_time=entry.recording_time,
            collection=entry.collection,
            session_key=_session_key(entry),
        )
        selected.append(
            with_recording_window(
                snippet,
                recordings_by_key.get(entry.source_recording_key)
                if entry.source_recording_key
                else None,
                start_offset_s=entry.hum_start_s,
                end_offset_s=entry.hum_end_s,
            )
        )

    return selected


def _copy_wav(src: Path, dst: Path, fs: FileSystem) -> None:
    fs.makedirs(dst.parent)
    in_handle = fs.open_read(src)
    try:
        out_handle = fs.open_write(dst)
        try:
            while True:
                chunk = in_handle.read(1024 * 1024)
                if not chunk:
                    break
                out_handle.write(chunk)
        finally:
            out_handle.close()
    finally:
        in_handle.close()
