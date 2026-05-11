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
from alpaca_pipelines.rf.audio_preprocess import prepare_rf_segment
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
    segment, rf_sr = prepare_rf_segment(
        signal=signal,
        source_sr=sample_rate,
        t0=start_s,
        t1=window_end_s,
        config=feature_config,
    )
    return {
        **raven_robust_features(
            y=segment,
            sr=rf_sr,
            fmin=feature_config.fmin_hz,
            fmax=feature_config.fmax_hz,
            n_fft=feature_config.n_fft,
            hop_length=feature_config.hop_length,
        ),
        **mfcc_summary(
            y=segment,
            sr=rf_sr,
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


def test_training_and_inference_feature_columns_still_match(tmp_path: Path) -> None:
    audio_path = tmp_path / "snippet.wav"
    _write_test_audio(audio_path)
    feature_config = RfFeatureConfig()

    training_features = rf_training_executor._compute_features_for_file(
        audio_path,
        feature_config=feature_config,
    )

    inference_features = _feature_row(audio_file=audio_path, feature_config=feature_config)

    assert list(training_features.keys()) == list(inference_features.keys())


def test_prepare_rf_segment_resamples_to_configured_sample_rate(tmp_path: Path) -> None:
    audio_path = tmp_path / "resample.wav"
    _write_test_audio(audio_path, sample_rate=16_000, duration_s=0.5)
    signal, source_sr = _mono_signal(audio_path)
    config = RfFeatureConfig(sample_rate_hz=48_000, min_duration_s=0.4)

    segment, rf_sr = prepare_rf_segment(
        signal=signal,
        source_sr=source_sr,
        t0=0.0,
        t1=0.5,
        config=config,
    )

    assert rf_sr == 48_000
    assert int(segment.shape[0]) == 24_000


def test_prepare_rf_segment_pads_short_segment_to_min_duration(tmp_path: Path) -> None:
    audio_path = tmp_path / "short-segment.wav"
    _write_test_audio(audio_path, sample_rate=48_000, duration_s=0.02)
    signal, source_sr = _mono_signal(audio_path)
    config = RfFeatureConfig(sample_rate_hz=48_000, min_duration_s=0.4, pad_short_segments=True)

    segment, _rf_sr = prepare_rf_segment(
        signal=signal,
        source_sr=source_sr,
        t0=0.0,
        t1=0.02,
        config=config,
    )

    assert int(segment.shape[0]) == 19_200


def test_mfcc_summary_short_padded_segment_produces_finite_delta_features(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"
    _write_test_audio(audio_path, sample_rate=48_000, duration_s=0.02)

    signal, source_sr = _mono_signal(audio_path)
    segment, rf_sr = prepare_rf_segment(
        signal=signal,
        source_sr=source_sr,
        t0=0.0,
        t1=0.02,
        config=RfFeatureConfig(sample_rate_hz=48_000, min_duration_s=0.4, include_deltas=True),
    )
    features = mfcc_summary(
        y=segment,
        sr=rf_sr,
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
        assert np.isfinite(features[f"d_mfcc{index}_mean"])
        assert np.isfinite(features[f"d_mfcc{index}_std"])
        assert np.isfinite(features[f"dd_mfcc{index}_mean"])
        assert np.isfinite(features[f"dd_mfcc{index}_std"])


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
    persisted_state = run_manager.load_state("rf_training", run_state.run_id)

    model_dir = Path(run_state.run_dir) / "outputs" / "model"
    metadata = read_json(model_dir / "rf_model_metadata.json")
    assert metadata["feature_family"] == "rf_v1"
    assert metadata["rf_threshold"] == 0.6
    assert metadata["feature_config"] == RfFeatureConfig().model_dump()
    assert metadata["feature_names"] == ["Dur 90% (s)", "mfcc1_mean"]

    report = read_json(
        Path(run_state.run_dir) / "outputs" / "summaries" / "rf_training_report.json"
    )
    assert report["feature_family"] == "rf_v1"
    assert report["rf_threshold"] == 0.6
    assert report["feature_config"] == RfFeatureConfig().model_dump()
    assert isinstance(report["metrics"]["roc_auc"], float)
    assert persisted_state.outputs.rf_training_report_path == str(
        Path(run_state.run_dir) / "outputs" / "summaries" / "rf_training_report.json"
    )
    assert persisted_state.progress.best_metric_name == "roc_auc"
    assert isinstance(persisted_state.progress.best_metric_value, float)


def test_rf_training_short_clip_with_deltas_completes_without_nan(
    monkeypatch: Any, tmp_path: Path
) -> None:
    environment = _environment(tmp_path)
    dataset_dir = environment.datasets_root / "dataset-short"
    dataset_dir.mkdir(parents=True)
    snippets_dir = tmp_path / "short-snippets"
    snippet_specs = [
        ("train_target.wav", "target", 440.0),
        ("train_noise.wav", "noise", 880.0),
        ("val_target.wav", "target", 660.0),
        ("val_noise.wav", "noise", 1320.0),
    ]
    for filename, _classification, frequency_hz in snippet_specs:
        _write_test_audio(
            snippets_dir / filename, sample_rate=48_000, duration_s=0.02, frequency_hz=frequency_hz
        )

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
    run_state = run_manager.create_run(
        "rf_training",
        {
            "dataset_name": "dataset-short",
            "feature_config": {
                "sample_rate_hz": 48_000,
                "min_duration_s": 0.4,
                "pad_short_segments": True,
                "include_deltas": True,
            },
        },
    )
    completed = rf_training_executor.execute_rf_training(run_state, environment, run_manager)
    assert completed.status == "completed"


def test_rf_training_uses_configured_threshold_for_validation_metrics(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    (environment.datasets_root / "dataset-a").mkdir(parents=True)
    run_manager = RunManager(environment.runs_root)
    run_state = run_manager.create_run(
        "rf_training",
        {"dataset_name": "dataset-a", "rf_threshold": 0.7},
    )

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
        labels = np.asarray([1, 0], dtype=np.int64)
        return table, labels, ["x.wav", "y.wav"]

    class _StubModel:
        feature_names_in_ = np.asarray(["Dur 90% (s)", "mfcc1_mean"], dtype=object)

        def fit(self, _x: pd.DataFrame, _y: np.ndarray) -> None:
            return

        def predict_proba(self, _x: pd.DataFrame) -> np.ndarray:
            return np.asarray([[0.35, 0.65], [0.45, 0.55]], dtype=np.float64)

    class _Logger:
        def info(self, _message: str) -> None:
            return

    monkeypatch.setattr(rf_training_executor, "_build_feature_table", _build_feature_table_stub)
    monkeypatch.setattr(
        rf_training_executor, "RandomForestClassifier", lambda **_kwargs: _StubModel()
    )
    monkeypatch.setattr(rf_training_executor, "create_logger", lambda *args, **kwargs: _Logger())
    monkeypatch.setattr(joblib, "dump", lambda _model, path: path.write_bytes(b""))

    completed = rf_training_executor.execute_rf_training(run_state, environment, run_manager)
    assert completed.status == "completed"

    report = read_json(
        Path(run_state.run_dir) / "outputs" / "summaries" / "rf_training_report.json"
    )
    assert report["rf_threshold"] == 0.7
    assert report["metrics"]["f1"] == 0.0
    assert report["metrics"]["roc_auc"] == 1.0


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

    rf_summary = apply_rf_filter(
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
    assert rf_summary["applied"] is True
    assert rf_summary["base_detections"] == 1
    assert rf_summary["rf_passed"] + rf_summary["rf_rejected"] + rf_summary["rf_unscored"] == 1
    assert len(rf_summary["files"]) == 1


def test_apply_rf_filter_returns_impact_summary(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(_PredictCalledModel(), model_path)
    write_json(
        model_dir / "rf_model_metadata.json",
        {
            "feature_family": "rf_v1",
            "feature_names": ["mfcc1_mean"],
            "feature_config": RfFeatureConfig().model_dump(),
        },
    )

    prediction_path = tmp_path / "predictions" / "audio.json"
    write_json(
        prediction_path,
        {
            "audio_file": str(audio_file),
            "n_windows": 0,
            "n_detections": 0,
            "detections": [],
            "scores_shape": [0, 0],
        },
    )
    summary = apply_rf_filter(
        prediction_inputs=[
            {"audio_file": str(audio_file), "prediction_file": str(prediction_path)}
        ],
        rf_model_path=str(model_path),
        rf_threshold=0.4,
        rf_feature_config=None,
        prediction_logger=logging.getLogger("test-rf-summary"),
    )
    assert summary["applied"] is True
    assert summary["rf_model_path"] == str(model_path)
    assert summary["base_detections"] == 0
    assert summary["rf_passed"] == 0
    assert summary["rf_rejected"] == 0
    assert summary["rf_unscored"] == 0
    assert summary["files"][0]["rf_filtered_file"].endswith("audio_rf_filtered.json")
    assert summary["files"][0]["rf_accepted_file"].endswith("audio_rf_accepted.json")
    assert summary["files"][0]["rf_rejected_file"].endswith("audio_rf_rejected.json")
    assert summary["files"][0]["rf_unscored_file"].endswith("audio_rf_unscored.json")


def test_rf_filter_writes_filtered_artifact_for_empty_detection_files(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(_PredictCalledModel(), model_path)
    write_json(
        model_dir / "rf_model_metadata.json",
        {
            "feature_family": "rf_v1",
            "feature_names": ["mfcc1_mean"],
            "feature_config": RfFeatureConfig().model_dump(),
        },
    )

    prediction_path = tmp_path / "predictions" / "audio.json"
    write_json(
        prediction_path,
        {
            "audio_file": str(audio_file),
            "n_windows": 0,
            "n_detections": 0,
            "detections": [],
            "scores_shape": [0, 0],
        },
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
        prediction_logger=logging.getLogger("test-rf-empty"),
    )

    filtered = read_json(prediction_path.with_name("audio_rf_filtered.json"))
    assert filtered["detections"] == []
    assert filtered["rf_filtered"] is True
    assert filtered["rf_model_path"] == str(model_path)
    assert filtered["rf_threshold"] == 0.4


def test_base_and_rf_filtered_outputs_both_remain_available(tmp_path: Path) -> None:
    audio_file = tmp_path / "collection_a" / "raw" / "same_stem.wav"
    _write_test_audio(audio_file)

    feature_config = RfFeatureConfig()
    feature_row = _feature_row(audio_file=audio_file, feature_config=feature_config)
    columns = list(feature_row.keys())
    training_table = pd.DataFrame(
        [feature_row, {k: float(v) + 1e-3 for k, v in feature_row.items()}],
        columns=columns,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=7)
    model.fit(training_table, np.asarray([0, 1], dtype=np.int64))
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
    _write_prediction_payload(prediction_path=prediction_path, audio_file=audio_file, end_s=0.4)

    apply_rf_filter(
        prediction_inputs=[
            {"audio_file": str(audio_file), "prediction_file": str(prediction_path)}
        ],
        rf_model_path=str(model_path),
        rf_threshold=0.4,
        rf_feature_config=None,
        prediction_logger=logging.getLogger("test-rf-preserve-base"),
    )

    assert prediction_path.is_file()
    assert prediction_path.with_name(f"{prediction_path.stem}_rf_filtered.json").is_file()
    assert prediction_path.with_name(f"{prediction_path.stem}_rf_accepted.json").is_file()
    assert prediction_path.with_name(f"{prediction_path.stem}_rf_rejected.json").is_file()
    assert prediction_path.with_name(f"{prediction_path.stem}_rf_unscored.json").is_file()


def test_rf_filter_summary_counts_pass_reject_unscored(monkeypatch: Any, tmp_path: Path) -> None:
    from alpaca_pipelines.rf import executor as rf_executor

    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)

    prediction_path = tmp_path / "predictions" / "audio.json"
    write_json(
        prediction_path,
        {
            "audio_file": str(audio_file),
            "n_windows": 3,
            "n_detections": 3,
            "detections": [
                {"start_s": 0.0, "end_s": 0.2, "score": 0.9},
                {"start_s": 0.2, "end_s": 0.4, "score": 0.8},
                {"start_s": 0.4, "end_s": 0.6, "score": 0.7},
            ],
            "scores_shape": [3, 2],
        },
    )

    class _Model:
        feature_names_in_ = np.asarray(["mfcc1_mean"], dtype=object)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            score = float(x[0, 0])
            return np.asarray([[1.0 - score, score]], dtype=np.float64)

    mfcc_calls = {"count": 0}

    def _mfcc_stub(**_kwargs: Any) -> dict[str, float]:
        mfcc_calls["count"] += 1
        if mfcc_calls["count"] == 1:
            return {"mfcc1_mean": 0.9}
        if mfcc_calls["count"] == 2:
            return {"mfcc1_mean": 0.1}
        return {"mfcc1_mean": float("nan")}

    monkeypatch.setattr(rf_executor, "_load_rf_model", lambda _path: _Model())
    monkeypatch.setattr(
        rf_executor, "_load_rf_model_metadata", lambda _path: {"feature_family": "rf_v1"}
    )
    monkeypatch.setattr(rf_executor, "_validate_rf_metadata_contract", lambda _metadata: None)
    monkeypatch.setattr(
        rf_executor,
        "_resolve_feature_config",
        lambda **_kwargs: RfFeatureConfig(),
    )
    read_calls = {"count": 0}

    def _read_segment_stub(
        _audio_handle: Any,
        *,
        start_s: float,
        end_s: float,
    ) -> tuple[np.ndarray, int]:
        read_calls["count"] += 1
        assert end_s >= start_s
        return np.zeros(16_000, dtype=np.float32), 16_000

    monkeypatch.setattr(rf_executor, "_read_audio_segment", _read_segment_stub)
    monkeypatch.setattr(
        rf_executor,
        "prepare_rf_segment",
        lambda **_kwargs: (np.zeros(4000, dtype=np.float32), 16_000),
    )
    monkeypatch.setattr(rf_executor, "raven_robust_features", lambda **_kwargs: {})
    monkeypatch.setattr(rf_executor, "mfcc_summary", _mfcc_stub)

    summary = apply_rf_filter(
        prediction_inputs=[
            {"audio_file": str(audio_file), "prediction_file": str(prediction_path)}
        ],
        rf_model_path=str(tmp_path / "model.joblib"),
        rf_threshold=0.4,
        rf_feature_config=None,
        prediction_logger=logging.getLogger("test-rf-counts"),
    )

    assert summary["base_detections"] == 3
    assert summary["rf_passed"] == 1
    assert summary["rf_rejected"] == 1
    assert summary["rf_unscored"] == 1
    assert summary["rejection_rate"] == 0.333333
    assert summary["pass_rate"] == 0.333333
    assert read_calls["count"] == 3
    file_summary = summary["files"][0]
    accepted_payload = read_json(Path(file_summary["rf_accepted_file"]))
    rejected_payload = read_json(Path(file_summary["rf_rejected_file"]))
    unscored_payload = read_json(Path(file_summary["rf_unscored_file"]))
    assert len(accepted_payload["detections"]) == 1
    assert len(rejected_payload["detections"]) == 1
    assert len(unscored_payload["detections"]) == 1


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


def test_rf_filter_uses_persisted_resample_and_padding_config(
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
    non_default_feature_config = RfFeatureConfig(
        sample_rate_hz=48_000,
        min_duration_s=0.4,
        pad_short_segments=True,
        n_fft=1024,
        hop_length=256,
        n_mfcc=8,
        include_deltas=True,
        fmin_hz=200.0,
        fmax_hz=3000.0,
    )
    run_state = run_manager.create_run(
        "rf_training",
        {
            "dataset_name": "dataset-b",
            "feature_config": non_default_feature_config.model_dump(),
        },
    )

    completed = rf_training_executor.execute_rf_training(run_state, environment, run_manager)
    assert completed.status == "completed"

    model_dir = Path(run_state.run_dir) / "outputs" / "model"
    model_path = model_dir / "rf_model.joblib"
    metadata = read_json(model_dir / "rf_model_metadata.json")
    assert metadata["feature_config"] == non_default_feature_config.model_dump()

    inference_audio = snippets_dir / "train_target.wav"

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    prediction_path = _prediction_output_path(predictions_dir, str(inference_audio))
    _write_prediction_payload(
        prediction_path=prediction_path, audio_file=inference_audio, end_s=0.02
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
    assert isinstance(filtered["detections"][0]["rf_pass"], bool)
