from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from alpaca_pipelines.datasets.audio_utils import concatenate_wavs
from alpaca_pipelines.datasets.contracts import Manifest
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.paths import (
    REVIEW_CONCAT_FILENAME,
    REVIEW_DIR,
    REVIEW_SELECTION_TABLE_FILENAME,
    SNIPPETS_DIR,
)


def prepare_review_artifacts(
    dataset_dir: Path,
    manifest: Manifest,
    gap_seconds: float,
    freq_low_hz: int,
    freq_high_hz: int,
    fs: FileSystem = _DEFAULT_FS,
) -> Path:
    review_dir = dataset_dir / REVIEW_DIR
    fs.makedirs(review_dir)

    snippets_dir = dataset_dir / SNIPPETS_DIR

    sorted_snippets = sorted(manifest.snippets, key=lambda s: (s.collection, s.uid))
    wav_paths = [snippets_dir / s.filename for s in sorted_snippets]

    for wav_path in wav_paths:
        if not fs.exists(wav_path):
            raise FileNotFoundError(f"Snippet wav not found: {wav_path}")

    concat_path = review_dir / REVIEW_CONCAT_FILENAME
    offsets = concatenate_wavs(wav_paths, concat_path, gap_seconds, fs=fs)

    selection_rows: list[dict[str, object]] = []
    for idx, (snippet, (begin_time, end_time)) in enumerate(
        zip(sorted_snippets, offsets, strict=True)
    ):
        selection_rows.append(
            {
                "Selection": idx + 1,
                "View": "Spectrogram 1",
                "Channel": 1,
                "Begin Time (s)": round(begin_time, 4),
                "End Time (s)": round(end_time, 4),
                "Low Freq (Hz)": freq_low_hz,
                "High Freq (Hz)": freq_high_hz,
                "Sound_type": snippet.classification,
                "uid": snippet.uid,
            }
        )

    selection_table = pd.DataFrame(selection_rows)

    buffer = io.StringIO()
    selection_table.to_csv(buffer, sep="\t", index=False)

    table_path = review_dir / REVIEW_SELECTION_TABLE_FILENAME
    fs.write_text(table_path, buffer.getvalue())

    return table_path
