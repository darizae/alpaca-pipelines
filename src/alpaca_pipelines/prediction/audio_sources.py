from __future__ import annotations

from pathlib import Path

from alpaca_pipelines.paths import validate_relative_path
from alpaca_pipelines.prediction.config import TapeFileHandleSpec

_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def resolve_tape_audio_files(
    *,
    collection_root: Path,
    tape_files: list[TapeFileHandleSpec],
) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()

    for handle in tape_files:
        relative_path = "{}/{}/{}".format(
            handle.collection_name,
            handle.category_dir,
            handle.relative_path,
        )
        resolved = validate_relative_path(relative_path, collection_root)
        if not resolved.is_file():
            raise FileNotFoundError("Tape file not found: {}".format(resolved))
        if resolved.suffix.lower() not in _AUDIO_SUFFIXES:
            raise ValueError("Unsupported tape file extension: {}".format(resolved))
        normalized = str(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        files.append(normalized)

    if not files:
        raise FileNotFoundError("No tape files resolved from tape_files handles")
    return files


def resolve_collection_audio_files(
    *,
    collection_root: Path,
    collection_names: list[str],
    source_category_dirs: list[str],
) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()

    for collection_name in collection_names:
        collection_dir = collection_root / collection_name
        if not collection_dir.is_dir():
            raise FileNotFoundError("Collection directory not found: {}".format(collection_dir))

        for category_dir in source_category_dirs:
            source_dir = collection_dir / category_dir
            if not source_dir.exists():
                continue
            if not source_dir.is_dir():
                raise ValueError("Source category path is not a directory: {}".format(source_dir))
            for candidate in sorted(source_dir.rglob("*"), key=lambda path: path.as_posix()):
                if not candidate.is_file() or candidate.suffix.lower() != ".wav":
                    continue
                resolved = str(candidate)
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(resolved)

    if not files:
        raise FileNotFoundError(
            "No .wav files found for collections {} in categories {} under {}".format(
                collection_names,
                source_category_dirs,
                collection_root,
            )
        )
    return files
