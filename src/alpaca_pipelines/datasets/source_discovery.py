from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from alpaca_pipelines.datasets.audio_utils import wav_duration_seconds
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.paths import find_collection_dirs, rglob_wavs


@dataclass(frozen=True)
class SourceAudioFile:
    path: Path
    relative_path: str
    collection: str
    duration_s: float


def discover_source_files(
    collection_root: Path,
    source_category_dirs: list[str],
    fs: FileSystem = _DEFAULT_FS,
) -> list[SourceAudioFile]:
    collection_dirs = find_collection_dirs(collection_root, fs)
    sources: list[SourceAudioFile] = []

    root_posix = PurePosixPath(collection_root.as_posix())

    for collection_dir in collection_dirs:
        for category_name in source_category_dirs:
            category_dir = collection_dir / category_name
            if not fs.exists(category_dir):
                continue
            if not fs.is_dir(category_dir):
                continue

            wav_files = rglob_wavs(category_dir, fs)
            for wav_path in wav_files:
                duration = wav_duration_seconds(wav_path, fs)

                wav_posix = PurePosixPath(wav_path.as_posix())
                try:
                    relative = str(wav_posix.relative_to(root_posix))
                except ValueError:
                    raise ValueError(
                        "Source wav path escapes collection root: {} (root={})".format(
                            wav_path, collection_root
                        )
                    ) from None

                sources.append(
                    SourceAudioFile(
                        path=wav_path,
                        relative_path=relative,
                        collection=collection_dir.name,
                        duration_s=duration,
                    )
                )

    if not sources:
        raise FileNotFoundError(
            f"No source audio files found in {source_category_dirs} under {collection_root}"
        )

    return sources
