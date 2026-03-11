from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.fs import FileSystem
from alpaca_pipelines.collections.parsing.patterns import (
    CANONICAL_CLIP_RE,
    COL1_CLIP_RE,
    COL2_LABELLED_DT_RE,
    COL2_LABELLED_SUBJECT_DATE_RE,
    COL2_LABELLED_SUBJECT_DT_RE,
)


@dataclass(frozen=True)
class NormalizedClipName:
    subject_alias: str
    subject_id: str
    date_yyyymmdd: str
    time_hhmmss: str | None
    note: str | None
    clip_start_s: int
    clip_end_s: int

    def to_filename(self) -> str:
        note_part = f"_{self.note}" if self.note else ""
        time_part = f"_{self.time_hhmmss}" if self.time_hhmmss else ""
        return (
            f"{self.subject_id}_{self.date_yyyymmdd}{time_part}{note_part}.wav_"
            f"{self.clip_start_s}_{self.clip_end_s}.wav"
        )


def wav_duration_seconds(path: Path, fs: FileSystem) -> float:
    try:
        fh = fs.open_read(path)
        try:
            info = sf.info(fh)
        finally:
            fh.close()
    except Exception as exc:
        raise ValueError(f"Failed to read wav header for duration: {path}") from exc

    frames = float(info.frames)
    samplerate = float(info.samplerate)
    if samplerate <= 0.0:
        raise ValueError(f"Invalid samplerate in wav header: {path} (samplerate={info.samplerate})")
    return frames / samplerate


def sanitize_note(note: str | None) -> str | None:
    if note is None:
        return None
    cleaned = note.strip().replace(" ", "_")
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or None


def normalize_labelled_clip_filename(
    path: Path,
    identity_map: IdentityMap,
    fs: FileSystem,
) -> NormalizedClipName:
    filename = path.name

    canonical_match = CANONICAL_CLIP_RE.match(filename)
    if canonical_match:
        subject_alias = canonical_match["subject"]
        subject_id = identity_map.normalize_subject(subject_alias)
        note = sanitize_note(canonical_match.group("note"))
        time_hhmmss = canonical_match.group("time")
        return NormalizedClipName(
            subject_alias=subject_alias,
            subject_id=subject_id,
            date_yyyymmdd=canonical_match["date"],
            time_hhmmss=time_hhmmss,
            note=note,
            clip_start_s=int(canonical_match["clip_start"]),
            clip_end_s=int(canonical_match["clip_end"]),
        )

    col1_match = COL1_CLIP_RE.match(filename)
    if col1_match:
        subject_alias = col1_match["subject"]
        subject_id = identity_map.normalize_subject(subject_alias)
        tag = col1_match.group("tag")
        note = sanitize_note(tag)
        return NormalizedClipName(
            subject_alias=subject_alias,
            subject_id=subject_id,
            date_yyyymmdd=col1_match["date"],
            time_hhmmss=None,
            note=note,
            clip_start_s=int(col1_match["clip_start"]),
            clip_end_s=int(col1_match["clip_end"]),
        )

    dt_match = COL2_LABELLED_DT_RE.match(filename)
    if dt_match:
        duration_s = int(round(wav_duration_seconds(path, fs)))
        note = sanitize_note(dt_match.group("note"))
        subject_alias = "UNKN"
        subject_id = identity_map.normalize_subject(subject_alias)
        return NormalizedClipName(
            subject_alias=subject_alias,
            subject_id=subject_id,
            date_yyyymmdd=dt_match["date"],
            time_hhmmss=dt_match["time"],
            note=note,
            clip_start_s=0,
            clip_end_s=duration_s,
        )

    subject_dt_match = COL2_LABELLED_SUBJECT_DT_RE.match(filename)
    if subject_dt_match:
        duration_s = int(round(wav_duration_seconds(path, fs)))
        note = sanitize_note(subject_dt_match.group("note"))
        subject_alias = subject_dt_match["subject"]
        subject_id = identity_map.normalize_subject(subject_alias)
        return NormalizedClipName(
            subject_alias=subject_alias,
            subject_id=subject_id,
            date_yyyymmdd=subject_dt_match["date"],
            time_hhmmss=subject_dt_match["time"],
            note=note,
            clip_start_s=0,
            clip_end_s=duration_s,
        )

    subject_date_match = COL2_LABELLED_SUBJECT_DATE_RE.match(filename)
    if subject_date_match:
        duration_s = int(round(wav_duration_seconds(path, fs)))
        note = sanitize_note(subject_date_match.group("note"))
        subject_alias = subject_date_match["subject"]
        subject_id = identity_map.normalize_subject(subject_alias)
        return NormalizedClipName(
            subject_alias=subject_alias,
            subject_id=subject_id,
            date_yyyymmdd=subject_date_match["date"],
            time_hhmmss=None,
            note=note,
            clip_start_s=0,
            clip_end_s=duration_s,
        )

    raise ValueError(f"Cannot normalize labelled clip filename: {filename}")
