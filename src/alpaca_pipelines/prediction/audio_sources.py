from __future__ import annotations

from pathlib import Path


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
