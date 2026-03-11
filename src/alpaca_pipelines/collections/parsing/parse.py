from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpaca_pipelines.collections.parsing.patterns import CANONICAL_CLIP_RE, CANONICAL_HUM_RE


@dataclass(frozen=True)
class ParsedHum:
    clip_filename: str
    hum_start_s: float
    hum_end_s: float
    source_quality: int


def _validate_quality(q: int, filename: str) -> int:
    if q not in (1, 2, 3, 4):
        raise ValueError(f"Invalid quality Q{q} in filename: {filename}")
    return q


def parse_canonical_hum_filename(filename: str) -> ParsedHum:
    match = CANONICAL_HUM_RE.match(filename)
    if not match:
        raise ValueError(f"Not a canonical hum filename: {filename}")
    q = _validate_quality(int(match["q"]), filename)
    return ParsedHum(
        clip_filename=match["clip"],
        hum_start_s=float(match["hum_start"]),
        hum_end_s=float(match["hum_end"]),
        source_quality=q,
    )


def normalize_segmented_hum_filename(hum_path: Path) -> str:
    """
    No inference, no mapping, no lookup.

    Accepted ONLY if:
      1) hum filename matches CANONICAL_HUM_RE, AND
      2) embedded clip filename matches CANONICAL_CLIP_RE (true canonical clip filename)
    """
    name = hum_path.name
    match = CANONICAL_HUM_RE.match(name)
    if not match:
        raise ValueError(
            "Non-canonical segmented hum filename encountered. "
            "Legacy hum naming is not supported; no clip mapping/inference exists. "
            f"Filename: {name}"
        )

    _validate_quality(int(match["q"]), name)

    clip_filename = match["clip"]
    if CANONICAL_CLIP_RE.match(clip_filename) is None:
        raise ValueError(
            "Hum embeds a non-canonical clip filename. "
            "Segmented hum filenames MUST embed a canonical clip filename. "
            f"Embedded clip: {clip_filename} (from {name})"
        )

    return name
