from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from alpaca_pipelines.collections.parsing.patterns import CANONICAL_CLIP_RE
from alpaca_pipelines.datasets.audio_utils import wav_duration_seconds
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.paths import find_collection_dirs, rglob_wavs
from alpaca_pipelines.recordings import (
    SourceRecording,
    derive_source_recording_key,
    load_collection_recordings,
)


@dataclass(frozen=True)
class SourceAudioFile:
    path: Path
    relative_path: str
    collection: str
    duration_s: float
    source_recording_key: str | None
    source_recording: SourceRecording | None
    clip_start_s: float | None
    clip_end_s: float | None


def discover_source_files(
    collection_root: Path,
    source_category_dirs: list[str],
    fs: FileSystem = _DEFAULT_FS,
) -> list[SourceAudioFile]:
    collection_dirs = find_collection_dirs(collection_root, fs)
    sources: list[SourceAudioFile] = []

    root_posix = PurePosixPath(collection_root.as_posix())

    for collection_dir in collection_dirs:
        collection_recordings = load_collection_recordings(collection_dir, fs)
        recordings_by_key = {recording.key: recording for recording in collection_recordings}
        for category_name in source_category_dirs:
            category_dir = collection_dir / category_name
            if not fs.exists(category_dir):
                continue
            if not fs.is_dir(category_dir):
                continue

            wav_files = rglob_wavs(category_dir, fs)
            for wav_path in wav_files:
                duration = wav_duration_seconds(wav_path, fs)
                source_recording_key, clip_start_s, clip_end_s = _recording_metadata_from_path(
                    wav_path
                )

                wav_posix = PurePosixPath(wav_path.as_posix())
                try:
                    relative = str(wav_posix.relative_to(root_posix))
                except ValueError:
                    raise ValueError(
                        "Source wav path escapes collection root: {} (root={})".format(
                            wav_path, collection_root
                        )
                    ) from None
                source_recording = (
                    recordings_by_key.get(source_recording_key)
                    if source_recording_key is not None
                    else None
                )

                sources.append(
                    SourceAudioFile(
                        path=wav_path,
                        relative_path=relative,
                        collection=collection_dir.name,
                        duration_s=duration,
                        source_recording_key=source_recording_key,
                        source_recording=source_recording,
                        clip_start_s=clip_start_s,
                        clip_end_s=clip_end_s,
                    )
                )

    if not sources:
        raise FileNotFoundError(
            f"No source audio files found in {source_category_dirs} under {collection_root}"
        )

    return sources


def _recording_metadata_from_path(path: Path) -> tuple[str | None, float | None, float | None]:
    match = CANONICAL_CLIP_RE.match(path.name)
    if match is None or match.group("time") is None:
        return None, None, None
    recording_key = derive_source_recording_key(
        subject_id=match["subject"],
        recording_date=f"{match['date'][0:4]}-{match['date'][4:6]}-{match['date'][6:8]}",
        recording_time=f"{match['time'][0:2]}:{match['time'][2:4]}:{match['time'][4:6]}",
    )
    return recording_key, float(match["clip_start"]), float(match["clip_end"])
