from __future__ import annotations

from pathlib import Path

import pytest

from alpaca_pipelines.prediction.audio_sources import (
    resolve_collection_audio_files,
    resolve_tape_audio_files,
)
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


def test_prediction_spec_accepts_tape_mode_with_tape_files() -> None:
    spec = PredictionRunSpec.model_validate(
        {
            "model_path": "/models/final.pt",
            "mode": "tape",
            "tape_files": [
                {
                    "collection_name": "audio_collection_388_m32_20250213",
                    "category_dir": "raw_recordings",
                    "relative_path": "nested/example.wav",
                }
            ],
        }
    )

    assert spec.mode == "tape"
    assert len(spec.tape_files) == 1
    assert spec.tape_files[0].collection_name == "audio_collection_388_m32_20250213"


def test_prediction_spec_rejects_legacy_audio_files_field() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PredictionRunSpec.model_validate(
            {
                "model_path": "/models/final.pt",
                "mode": "tape",
                "audio_files": [
                    "/collections/audio_collection_388_m32_20250213/raw_recordings/a.wav"
                ],
            }
        )


def test_resolve_tape_audio_files_resolves_handle_paths(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection-root"
    source_dir = collection_root / "audio_collection_388_m32_20250213" / "raw_recordings" / "nested"
    source_dir.mkdir(parents=True)
    (source_dir / "example.wav").write_bytes(b"")

    spec = PredictionRunSpec.model_validate(
        {
            "model_path": "/models/final.pt",
            "mode": "tape",
            "tape_files": [
                {
                    "collection_name": "audio_collection_388_m32_20250213",
                    "category_dir": "raw_recordings",
                    "relative_path": "nested/example.wav",
                }
            ],
        }
    )

    files = resolve_tape_audio_files(
        collection_root=collection_root,
        tape_files=spec.tape_files,
    )

    assert files == [str(source_dir / "example.wav")]


def test_resolve_tape_audio_files_rejects_missing_path(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection-root"
    (collection_root / "audio_collection_388_m32_20250213" / "raw_recordings").mkdir(parents=True)
    spec = PredictionRunSpec.model_validate(
        {
            "model_path": "/models/final.pt",
            "mode": "tape",
            "tape_files": [
                {
                    "collection_name": "audio_collection_388_m32_20250213",
                    "category_dir": "raw_recordings",
                    "relative_path": "missing.wav",
                }
            ],
        }
    )

    with pytest.raises(FileNotFoundError, match="Tape file not found"):
        resolve_tape_audio_files(
            collection_root=collection_root,
            tape_files=spec.tape_files,
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


def test_prediction_spec_dataset_mode_requires_rf_model_path_when_rf_filter_enabled() -> None:
    with pytest.raises(
        ValueError,
        match="rf_model_path is required when apply_rf_filter is enabled",
    ):
        PredictionRunSpec(
            model_path="/models/final.pt",
            mode="dataset",
            dataset_name="dataset_a",
            apply_rf_filter=True,
        )


def test_prediction_spec_accepts_upstream_run_provenance_fields() -> None:
    spec = PredictionRunSpec(
        training_run_id="training-run-1",
        model_path="/models/final.pt",
        mode="dataset",
        dataset_name="dataset_a",
        apply_rf_filter=True,
        rf_training_run_id="rf-training-run-1",
        rf_model_path="/models/rf.joblib",
    )

    assert spec.training_run_id == "training-run-1"
    assert spec.rf_training_run_id == "rf-training-run-1"


def test_prediction_spec_rejects_rf_training_run_id_when_rf_filter_disabled() -> None:
    with pytest.raises(
        ValueError,
        match="rf_training_run_id must be null when apply_rf_filter is disabled",
    ):
        PredictionRunSpec(
            training_run_id="training-run-1",
            model_path="/models/final.pt",
            mode="dataset",
            dataset_name="dataset_a",
            rf_training_run_id="rf-training-run-1",
        )
