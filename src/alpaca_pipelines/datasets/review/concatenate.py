from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import soundfile as sf

from alpaca_pipelines.datasets.audio_utils import concatenate_wavs
from alpaca_pipelines.datasets.contracts import Manifest
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.paths import (
    REVIEW_DIR,
    REVIEW_NOISE_CONCAT_FILENAME,
    REVIEW_NOISE_SELECTION_TABLE_FILENAME,
    REVIEW_TARGET_CONCAT_FILENAME,
    REVIEW_TARGET_SELECTION_TABLE_FILENAME,
    SNIPPETS_DIR,
)

ReviewClass = Literal["target", "noise"]
_REVIEW_CLASSES: tuple[ReviewClass, ...] = ("target", "noise")
_REVIEW_CONCAT_FILENAMES: dict[ReviewClass, str] = {
    "target": REVIEW_TARGET_CONCAT_FILENAME,
    "noise": REVIEW_NOISE_CONCAT_FILENAME,
}
_REVIEW_SELECTION_TABLE_FILENAMES: dict[ReviewClass, str] = {
    "target": REVIEW_TARGET_SELECTION_TABLE_FILENAME,
    "noise": REVIEW_NOISE_SELECTION_TABLE_FILENAME,
}


@dataclass(frozen=True)
class ReviewClassArtifacts:
    concat_wav_path: Path
    selection_table_path: Path
    n_snippets: int


def prepare_review_artifacts(
    dataset_dir: Path,
    manifest: Manifest,
    gap_seconds: float,
    freq_low_hz: int,
    freq_high_hz: int,
    fs: FileSystem = _DEFAULT_FS,
) -> dict[ReviewClass, ReviewClassArtifacts]:
    review_dir = dataset_dir / REVIEW_DIR
    fs.makedirs(review_dir)
    snippets_dir = dataset_dir / SNIPPETS_DIR
    results: dict[ReviewClass, ReviewClassArtifacts] = {}

    for review_class in _REVIEW_CLASSES:
        class_snippets = sorted(
            [snippet for snippet in manifest.snippets if snippet.classification == review_class],
            key=lambda snippet: (snippet.collection, snippet.uid),
        )
        wav_paths = [snippets_dir / snippet.filename for snippet in class_snippets]
        for wav_path in wav_paths:
            if not fs.exists(wav_path):
                raise FileNotFoundError(f"Snippet wav not found: {wav_path}")

        concat_path = review_dir / _REVIEW_CONCAT_FILENAMES[review_class]
        if wav_paths:
            offsets = concatenate_wavs(wav_paths, concat_path, gap_seconds, fs=fs)
        else:
            _write_empty_wav(concat_path, fs)
            offsets = []

        selection_rows: list[dict[str, object]] = []
        for idx, (snippet, (begin_time, end_time)) in enumerate(
            zip(class_snippets, offsets, strict=True)
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
                    "review_label": snippet.classification,
                    "uid": snippet.uid,
                }
            )

        selection_table = pd.DataFrame(selection_rows)
        buffer = io.StringIO()
        selection_table.to_csv(buffer, sep="\t", index=False)
        selection_table_path = review_dir / _REVIEW_SELECTION_TABLE_FILENAMES[review_class]
        fs.write_text(selection_table_path, buffer.getvalue())

        results[review_class] = ReviewClassArtifacts(
            concat_wav_path=concat_path,
            selection_table_path=selection_table_path,
            n_snippets=len(class_snippets),
        )

    return results


def _write_empty_wav(path: Path, fs: FileSystem) -> None:
    handle = fs.open_write(path)
    try:
        sf.write(handle, np.array([], dtype=np.float32), 16000)
    finally:
        handle.close()
