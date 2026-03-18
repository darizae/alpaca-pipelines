from __future__ import annotations

from pathlib import Path

import pytest
from alpaca_pipelines.prediction.audio_sources import resolve_collection_audio_files
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.prediction.executor import _prediction_output_path


def test_prediction_spec_accepts_collection_mode() -> None:
    spec = PredictionRunSpec(
        model_path="/models/final.pt",
        mode="collection",
        collection_names=["audio_collection_388_m32_20250213"],
        source_category_dirs=["raw_recordings"],
    )

    assert spec.mode == "collection"
    assert spec.collection_names == ["audio_collection_388_m32_20250213"]
    assert spec.source_category_dirs == ["raw_recordings"]


def test_prediction_spec_collection_mode_requires_collection_names() -> None:
    with pytest.raises(ValueError, match="collection_names are required for collection mode"):
        PredictionRunSpec(
            model_path="/models/final.pt",
            mode="collection",
            source_category_dirs=["raw_recordings"],
        )


def test_prediction_spec_collection_mode_rejects_invalid_collection_name() -> None:
    with pytest.raises(
        ValueError,
        match="collection_names entries must start with 'audio_collection_'",
    ):
        PredictionRunSpec(
            model_path="/models/final.pt",
            mode="collection",
            collection_names=["raw_batch_388"],
            source_category_dirs=["raw_recordings"],
        )


def test_resolve_collection_audio_files_finds_wav_files(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection-root"
    raw_dir = collection_root / "audio_collection_388_m32_20250213" / "raw_recordings"
    nested = raw_dir / "nested"
    nested.mkdir(parents=True)
    (raw_dir / "a.WAV").write_bytes(b"")
    (nested / "b.wav").write_bytes(b"")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")

    files = resolve_collection_audio_files(
        collection_root=collection_root,
        collection_names=["audio_collection_388_m32_20250213"],
        source_category_dirs=["raw_recordings"],
    )

    assert files == [
        str(raw_dir / "a.WAV"),
        str(nested / "b.wav"),
    ]


def test_resolve_collection_audio_files_fails_when_empty(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection-root"
    (collection_root / "audio_collection_388_m32_20250213" / "raw_recordings").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .wav files found"):
        resolve_collection_audio_files(
            collection_root=collection_root,
            collection_names=["audio_collection_388_m32_20250213"],
            source_category_dirs=["raw_recordings"],
        )


def test_prediction_output_path_is_unique_for_same_stem_different_paths() -> None:
    predictions_dir = Path("/tmp/predictions")
    first = _prediction_output_path(
        predictions_dir,
        "/root/audio_collection_1/raw_recordings/20250211_075558.WAV",
    )
    second = _prediction_output_path(
        predictions_dir,
        "/root/audio_collection_2/raw_recordings/20250211_075558.WAV",
    )

    assert first != second
