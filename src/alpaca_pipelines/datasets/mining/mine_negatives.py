from __future__ import annotations

import random
from itertools import count
from pathlib import Path

from alpaca_pipelines.datasets.audio_utils import extract_segment
from alpaca_pipelines.datasets.contracts import SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.recording_metadata import with_recording_window
from alpaca_pipelines.datasets.source_discovery import SourceAudioFile


def _snippet_filename(label: str, uid: int, collection: str) -> str:
    return f"noise-{label}_{uid:06d}_{collection}.wav"


def _pick_random_position(
    source: SourceAudioFile,
    desired_duration: float,
    rng: random.Random,
) -> tuple[float, float] | None:
    available = source.duration_s - desired_duration
    if available < 0.0:
        return None
    start = rng.uniform(0.0, max(available, 0.0))
    return (start, start + desired_duration)


def mine_negatives_for_positives(
    positive_snippets: list[SnippetEntry],
    source_files: list[SourceAudioFile],
    noise_per_positive: float,
    attempts_per_slot: int,
    snippets_dir: Path,
    uid_counter: count[int],
    seed: int,
    fs: FileSystem = _DEFAULT_FS,
) -> list[SnippetEntry]:
    rng = random.Random(seed)

    if not source_files:
        raise ValueError("No source audio files available for mining negatives")

    mined: list[SnippetEntry] = []
    slots_requested = 0
    slots_filled = 0

    for positive in positive_snippets:
        target_duration = positive.duration_s

        noise_count_whole = int(noise_per_positive)
        fractional = noise_per_positive - noise_count_whole
        if rng.random() < fractional:
            noise_count_whole += 1

        for _ in range(noise_count_whole):
            slots_requested += 1
            snippet_entry = _mine_single_slot(
                target_duration=target_duration,
                source_files=source_files,
                attempts_per_slot=attempts_per_slot,
                snippets_dir=snippets_dir,
                uid_counter=uid_counter,
                rng=rng,
                fs=fs,
            )
            if snippet_entry is not None:
                mined.append(snippet_entry)
                slots_filled += 1

    if slots_filled < slots_requested:
        raise ValueError(
            f"Noise mining underfill: requested {slots_requested} slots, "
            f"filled {slots_filled}. "
            f"{slots_requested - slots_filled} slots failed after "
            f"{attempts_per_slot} attempts each. "
            f"Check source file durations vs. target snippet durations."
        )

    return mined


def _mine_single_slot(
    target_duration: float,
    source_files: list[SourceAudioFile],
    attempts_per_slot: int,
    snippets_dir: Path,
    uid_counter: count[int],
    rng: random.Random,
    fs: FileSystem,
) -> SnippetEntry | None:
    for _ in range(attempts_per_slot):
        source = rng.choice(source_files)
        position = _pick_random_position(source, target_duration, rng)
        if position is None:
            continue

        start_s, end_s = position
        uid = next(uid_counter)
        filename = _snippet_filename("bg", uid, source.collection)
        destination = snippets_dir / filename

        try:
            actual_duration = extract_segment(
                source.path,
                start_s,
                end_s,
                destination,
                fs=fs,
            )
        except Exception:
            if fs.exists(destination):
                fs.unlink(destination)
            continue

        snippet = SnippetEntry(
            uid=uid,
            filename=filename,
            classification="noise",
            source_type="mined_source",
            source_path=source.relative_path,
            start_s=round(start_s, 4),
            end_s=round(end_s, 4),
            duration_s=round(actual_duration, 4),
            quality=None,
            subject_id=None,
            recording_date=None,
            collection=source.collection,
            session_key=None,
        )
        source_recording_start_s = (
            source.clip_start_s + start_s if source.clip_start_s is not None else None
        )
        source_recording_end_s = (
            source.clip_start_s + end_s if source.clip_start_s is not None else None
        )
        return with_recording_window(
            snippet,
            source.source_recording,
            start_offset_s=source_recording_start_s,
            end_offset_s=source_recording_end_s,
        )

    return None
