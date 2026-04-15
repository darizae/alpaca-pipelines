from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.executor import _prediction_output_path
from alpaca_pipelines.rf.audio_features import mfcc_summary, raven_robust_features
from alpaca_pipelines.rf.config import RfFeatureConfig
from alpaca_pipelines.rf.executor import apply_rf_filter
from alpaca_pipelines.rf_training import executor as rf_training_executor
from alpaca_pipelines.runs.manager import RunManager


class _PredictCalledModel:
    def predict_proba(self, _feature_vector: np.ndarray) -> np.ndarray:
        raise AssertionError("predict_proba must not be called")


def _write_test_audio(
    path: Path,
    sample_rate: int = 16_000,
    duration_s: float = 1.5,
    frequency_hz: float = 440.0,
) -> None:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = (0.2 * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate)


def _mono_signal(path: Path) -> tuple[np.ndarray, int]:
    signal, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if signal.shape[1] > 1:
        signal = np.mean(signal, axis=1)
    else:
        signal = signal[:, 0]
    return signal, int(sample_rate)


def _environment(tmp_path: Path) -> PipelineEnvironment:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir()
    datasets_root.mkdir()
    write_json(collection_root / "merged_index.json", {"meta": {}, "entries": []})
    return PipelineEnvironment.from_explicit(
        collection_root=collection_root,
        merged_index_path=collection_root / "merged_index.json",
        datasets_root=datasets_root,
        runs_root=runs_root,
    )


def _feature_row(
    audio_file: Path,
    feature_config: RfFeatureConfig,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> dict[str, float]:
    signal, sample_rate = _mono_signal(audio_file)
    window_end_s = float(len(signal)) / float(sample_rate) if end_s is None else end_s
    return {
        **raven_robust_features(
            y=signal,
            sr=sample_rate,
            t0=start_s,
            t1=window_end_s,
            fmin=feature_config.fmin_hz,
            fmax=feature_config.fmax_hz,
            n_fft=feature_config.n_fft,
            hop_length=feature_config.hop_length,
        ),
        **mfcc_summary(
            y=signal,
            sr=sample_rate,
            t0=start_s,
            t1=window_end_s,
            n_mfcc=feature_config.n_mfcc,
            n_fft=feature_config.n_fft,
            hop_length=feature_config.hop_length,
            include_deltas=feature_config.include_deltas,
        ),
    }


def _write_prediction_payload(prediction_path: Path, audio_file: Path, end_s: float) -> None:
    write_json(
        prediction_path,
        {
            "audio_file": str(audio_file),
            "n_windows": 3,
            "n_detections": 1,
            "detections": [{"start_s": 0.0, "end_s": end_s, "score": 0.9}],
            "scores_shape": [3, 2],
        },
    )


def test_shared_feature_columns_match_between_training_and_inference_paths(tmp_path: Path) -> None:
    audio_path = tmp_path / "snippet.wav"
    _write_test_audio(audio_path)
    feature_config = RfFeatureConfig()

    training_features = rf_training_executor._compute_features_for_file(
        audio_path,
        feature_config=feature_config,
    )

    inference_features = _feature_row(audio_file=audio_path, feature_config=feature_config)

    assert list(training_features.keys()) == list(inference_features.keys())


def test_mfcc_summary_short_interval_fills_delta_features_with_nan(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"
    _write_test_audio(audio_path, duration_s=0.02)

    signal, sample_rate = _mono_signal(audio_path)
    features = mfcc_summary(
        y=signal,
        sr=sample_rate,
        t0=0.0,
        t1=0.02,
        n_mfcc=13,
        n_fft=2048,
        hop_length=1024,
        include_deltas=True,
    )

    for index in range(1, 14):
        assert f"mfcc{index}_mean" in features
        assert f"mfcc{index}_std" in features
        assert f"d_mfcc{index}_mean" in features
        assert f"d_mfcc{index}_std" in features
        assert f"dd_mfcc{index}_mean" in features
        assert f"dd_mfcc{index}_std" in features
        assert np.isnan(features[f"d_mfcc{index}_mean"])
        assert np.isnan(features[f"d_mfcc{index}_std"])
        assert np.isnan(features[f"dd_mfcc{index}_mean"])
        assert np.isnan(features[f"dd_mfcc{index}_std"])


def test_rf_training_persists_feature_config_metadata(monkeypatch: Any, tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    (environment.datasets_root / "dataset-a").mkdir(parents=True)
    run_manager = RunManager(environment.runs_root)
    run_state = run_manager.create_run("rf_training", {"dataset_name": "dataset-a"})

    monkeypatch.setattr(
        rf_training_executor,
        "load_dataset_handle",
        lambda *_args, **_kwargs: SimpleNamespace(
            splits=SimpleNamespace(train=["a.wav", "b.wav"], val=["c.wav", "d.wav"]),
            class_to_index={"noise": 0, "target": 1},
            classes=["noise", "target"],
        ),
    )

    def _build_feature_table_stub(**_kwargs: Any) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        table = pd.DataFrame(
            [
                {"Dur 90% (s)": 0.1, "mfcc1_mean": 1.0},
                {"Dur 90% (s)": 0.2, "mfcc1_mean": 2.0},
            ]
        )
        labels = np.asarray([0, 1], dtype=np.int64)
        return table, labels, ["x.wav", "y.wav"]

    monkeypatch.setattr(rf_training_executor, "_build_feature_table", _build_feature_table_stub)

    class _Logger:
        def info(self, _message: str) -> None:
            return

    monkeypatch.setattr(rf_training_executor, "create_logger", lambda *args, **kwargs: _Logger())

    completed = rf_training_executor.execute_rf_training(run_state, environment, run_manager)
    assert completed.status == "completed"

    model_dir = Path(run_state.run_dir) / "outputs" / "model"
    metadata = read_json(model_dir / "rf_model_metadata.json")
    assert metadata["feature_family"] == "rf_v1"
    assert metadata["feature_config"] == RfFeatureConfig().model_dump()
    assert metadata["feature_names"] == ["Dur 90% (s)", "mfcc1_mean"]

    report = read_json(
        Path(run_state.run_dir) / "outputs" / "summaries" / "rf_training_report.json"
    )
    assert report["feature_family"] == "rf_v1"
    assert report["feature_config"] == RfFeatureConfig().model_dump()


def test_rf_filter_reads_exact_prediction_paths_from_prediction_executor(tmp_path: Path) -> None:
    audio_file = tmp_path / "collection_a" / "raw" / "same_stem.wav"
    _write_test_audio(audio_file)

    feature_config = RfFeatureConfig()
    signal, sample_rate = _mono_signal(audio_file)
    duration_s = float(len(signal)) / float(sample_rate)
    feature_row = _feature_row(audio_file=audio_file, feature_config=feature_config)
    columns = list(feature_row.keys())
    training_table = pd.DataFrame(
        [feature_row, {k: float(v) + 1e-3 for k, v in feature_row.items()}],
        columns=columns,
    )
    labels = np.asarray([0, 1], dtype=np.int64)

    model = RandomForestClassifier(n_estimators=10, random_state=7)
    model.fit(training_table, labels)

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(model, model_path)
    write_json(
        model_dir / "rf_model_metadata.json",
        {
            "feature_family": "rf_v1",
            "feature_names": columns,
            "feature_config": feature_config.model_dump(),
        },
    )

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    prediction_path = _prediction_output_path(predictions_dir, str(audio_file))
    _write_prediction_payload(
        prediction_path=prediction_path,
        audio_file=audio_file,
        end_s=duration_s,
    )

    apply_rf_filter(
        prediction_inputs=[
            {
                "audio_file": str(audio_file),
                "prediction_file": str(prediction_path),
            }
        ],
        rf_model_path=str(model_path),
        rf_threshold=0.4,
        rf_feature_config=None,
        prediction_logger=logging.getLogger("test-rf-filter"),
    )

    filtered_path = prediction_path.with_name(f"{prediction_path.stem}_rf_filtered.json")
    filtered = read_json(filtered_path)
    assert filtered["rf_filtered"] is True
    assert filtered["detections"][0]["rf_score"] is not None
    assert isinstance(filtered["detections"][0]["rf_pass"], bool)


def test_rf_filter_rejects_legacy_cnn_logit_models(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    signal, sample_rate = _mono_signal(audio_file)
    duration_s = float(len(signal)) / float(sample_rate)

    model = RandomForestClassifier(n_estimators=5, random_state=3)
    model.fit(
        pd.DataFrame(
            [
                {"cnn_logit_mean": 0.1, "mfcc1_mean": 1.0},
                {"cnn_logit_mean": 0.9, "mfcc1_mean": 2.0},
            ]
        ),
        np.asarray([0, 1], dtype=np.int64),
    )

    model_dir = tmp_path / "legacy_model"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(model, model_path)
    write_json(
        model_dir / "rf_model_metadata.json",
        {
            "feature_family": "rf_v1",
            "feature_config": RfFeatureConfig().model_dump(),
        },
    )

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    prediction_path = _prediction_output_path(predictions_dir, str(audio_file))
    _write_prediction_payload(
        prediction_path=prediction_path,
        audio_file=audio_file,
        end_s=duration_s,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported RF model: legacy feature 'cnn_logit_mean' is not supported by rf_v1",
    ):
        apply_rf_filter(
            prediction_inputs=[
                {
                    "audio_file": str(audio_file),
                    "prediction_file": str(prediction_path),
                }
            ],
            rf_model_path=str(model_path),
            rf_threshold=0.4,
            rf_feature_config=None,
            prediction_logger=logging.getLogger("test-rf-legacy"),
        )


def test_rf_filter_rejects_non_rf_v1_feature_family(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    signal, sample_rate = _mono_signal(audio_file)
    duration_s = float(len(signal)) / float(sample_rate)

    model_dir = tmp_path / "invalid_family"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(_PredictCalledModel(), model_path)
    write_json(
        model_dir / "rf_model_metadata.json",
        {
            "feature_family": "legacy_v0",
            "feature_names": ["mfcc1_mean"],
            "feature_config": RfFeatureConfig().model_dump(),
        },
    )

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    prediction_path = _prediction_output_path(predictions_dir, str(audio_file))
    _write_prediction_payload(
        prediction_path=prediction_path,
        audio_file=audio_file,
        end_s=duration_s,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported RF model feature_family",
    ):
        apply_rf_filter(
            prediction_inputs=[
                {
                    "audio_file": str(audio_file),
                    "prediction_file": str(prediction_path),
                }
            ],
            rf_model_path=str(model_path),
            rf_threshold=0.4,
            rf_feature_config=None,
            prediction_logger=logging.getLogger("test-rf-family"),
        )


def test_rf_training_to_inference_metadata_round_trip_uses_persisted_feature_config(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    dataset_dir = environment.datasets_root / "dataset-b"
    dataset_dir.mkdir(parents=True)
    snippets_dir = tmp_path / "snippets"

    snippet_specs = [
        ("train_target.wav", "target", 440.0),
        ("train_noise.wav", "noise", 880.0),
        ("val_target.wav", "target", 660.0),
        ("val_noise.wav", "noise", 1320.0),
    ]
    for filename, _, frequency_hz in snippet_specs:
        _write_test_audio(snippets_dir / filename, duration_s=1.0, frequency_hz=frequency_hz)

    dataset_handle = SimpleNamespace(
        snippets_dir=snippets_dir,
        splits=SimpleNamespace(
            train=["train_target.wav", "train_noise.wav"],
            val=["val_target.wav", "val_noise.wav"],
        ),
        class_to_index={"noise": 0, "target": 1},
        classes=["noise", "target"],
        manifest=SimpleNamespace(
            snippets=[
                SimpleNamespace(filename=filename, classification=classification)
                for filename, classification, _ in snippet_specs
            ]
        ),
    )
    monkeypatch.setattr(
        rf_training_executor,
        "load_dataset_handle",
        lambda *_args, **_kwargs: dataset_handle,
    )

    run_manager = RunManager(environment.runs_root)
    non_default_feature_config = {
        "n_fft": 1024,
        "hop_length": 256,
        "n_mfcc": 8,
        "include_deltas": True,
        "fmin_hz": 200.0,
        "fmax_hz": 3000.0,
    }
    run_state = run_manager.create_run(
        "rf_training",
        {
            "dataset_name": "dataset-b",
            "feature_config": non_default_feature_config,
        },
    )

    completed = rf_training_executor.execute_rf_training(run_state, environment, run_manager)
    assert completed.status == "completed"

    model_dir = Path(run_state.run_dir) / "outputs" / "model"
    model_path = model_dir / "rf_model.joblib"
    metadata = read_json(model_dir / "rf_model_metadata.json")
    assert metadata["feature_config"] == non_default_feature_config

    inference_audio = snippets_dir / "train_target.wav"
    signal, sample_rate = _mono_signal(inference_audio)
    duration_s = float(len(signal)) / float(sample_rate)

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    prediction_path = _prediction_output_path(predictions_dir, str(inference_audio))
    _write_prediction_payload(
        prediction_path=prediction_path,
        audio_file=inference_audio,
        end_s=duration_s,
    )

    apply_rf_filter(
        prediction_inputs=[
            {
                "audio_file": str(inference_audio),
                "prediction_file": str(prediction_path),
            }
        ],
        rf_model_path=str(model_path),
        rf_threshold=0.4,
        rf_feature_config=None,
        prediction_logger=logging.getLogger("test-rf-round-trip"),
    )

    filtered = read_json(prediction_path.with_name(f"{prediction_path.stem}_rf_filtered.json"))
    assert filtered["detections"][0]["rf_score"] is not None
