from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction import executor as prediction_executor
from alpaca_pipelines.runs.manager import RunManager


def _build_environment(tmp_path: Path) -> PipelineEnvironment:
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


def _prediction_spec(*, apply_rf_filter: bool = False) -> dict[str, Any]:
    return {
        "model_path": "/models/final.pt",
        "mode": "tape",
        "tape_files": [
            {
                "collection_name": "audio_collection_388_m32_20250213",
                "category_dir": "raw_recordings",
                "relative_path": "a.wav",
            }
        ],
        "run_name": "prediction-progress-test",
        "apply_rf_filter": apply_rf_filter,
        "rf_model_path": "/models/rf.joblib" if apply_rf_filter else None,
    }


def _install_prediction_stubs(
    monkeypatch: Any,
    *,
    with_rf_filter: bool,
) -> list[str]:
    class _Logger:
        def info(self, _message: str) -> None:
            return

    monkeypatch.setattr(prediction_executor, "create_logger", lambda *args, **kwargs: _Logger())
    monkeypatch.setattr(
        prediction_executor,
        "_load_trained_model",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(sample_rate=1000, min_level_db=-100, ref_level_db=20),
            SimpleNamespace(),
            {"noise": 0, "target": 1},
        ),
    )
    monkeypatch.setattr(
        prediction_executor,
        "resolve_tape_audio_files",
        lambda **_kwargs: ["/audio/a.wav", "/audio/b.wav"],
    )
    monkeypatch.setattr(
        prediction_executor,
        "_predict_tape",
        lambda *args, **kwargs: _fake_predict_tape(*args, **kwargs),
    )
    monkeypatch.setattr(
        prediction_executor,
        "_generate_detections",
        lambda tape_result, **_kwargs: (
            [
                {"start_s": 0.0, "end_s": 0.2, "score": 0.91},
                {"start_s": 0.3, "end_s": 0.5, "score": 0.93},
            ]
            if str(tape_result["audio_file"]).endswith("a.wav")
            else [{"start_s": 0.1, "end_s": 0.2, "score": 0.88}]
        ),
    )

    rf_calls: list[str] = []
    if with_rf_filter:
        monkeypatch.setattr(
            "alpaca_pipelines.rf.executor.apply_rf_filter",
            lambda **_kwargs: rf_calls.append("called"),
        )
    return rf_calls


def _fake_predict_tape(
    *,
    audio_file: str,
    hop_samples: int,
    sequence_length_samples: int,
    spec_config: Any,
    progress_callback: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback(0, 4)
        progress_callback(1, 4)
        progress_callback(4, 4)
    return {
        "audio_file": audio_file,
        "n_windows": 4,
        "hop_samples": hop_samples,
        "sequence_length_samples": sequence_length_samples,
        "sample_rate": spec_config.sample_rate,
        "scores": [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
    }


def _stage_sequence(progress_snapshots: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []
    for item in progress_snapshots:
        stage = str(item["stage"])
        if not sequence or sequence[-1] != stage:
            sequence.append(stage)
    return sequence


def test_prediction_progress_snapshot_contract(monkeypatch: Any, tmp_path: Path) -> None:
    environment = _build_environment(tmp_path)
    run_manager = RunManager(environment.runs_root)
    run_state = run_manager.create_run("prediction", _prediction_spec())
    _install_prediction_stubs(monkeypatch, with_rf_filter=False)

    progress_snapshots: list[dict[str, Any]] = []
    original_update_progress = run_manager.update_progress

    def _record_progress(*args: Any, **kwargs: Any) -> Any:
        state = original_update_progress(*args, **kwargs)
        snapshot = state.progress.prediction
        if snapshot is not None:
            progress_snapshots.append(snapshot.model_dump())
        return state

    monkeypatch.setattr(run_manager, "update_progress", _record_progress)

    completed = prediction_executor.execute_prediction(run_state, environment, run_manager)

    assert completed.status == "completed"
    persisted = read_json(Path(run_state.run_dir) / "run_state.json")
    assert persisted["progress"]["current_epoch"] is None
    assert persisted["progress"]["total_epochs"] is None
    prediction_progress = persisted["progress"]["prediction"]
    assert prediction_progress["stage"] == "completed"
    assert prediction_progress["files_total"] == 2
    assert prediction_progress["files_completed"] == 2
    assert prediction_progress["detections_so_far"] == 3
    assert prediction_progress["current_file"] is None
    assert prediction_progress["current_file_windows_total"] is None
    assert prediction_progress["current_file_windows_completed"] is None
    assert isinstance(prediction_progress["updated_at"], str)
    assert _stage_sequence(progress_snapshots) == [
        "initializing",
        "resolving_inputs",
        "predicting",
        "writing_summary",
        "completed",
    ]


def test_prediction_progress_stage_includes_rf_filtering(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    environment = _build_environment(tmp_path)
    run_manager = RunManager(environment.runs_root)
    run_state = run_manager.create_run(
        "prediction",
        _prediction_spec(apply_rf_filter=True),
    )
    rf_calls = _install_prediction_stubs(monkeypatch, with_rf_filter=True)

    progress_snapshots: list[dict[str, Any]] = []
    original_update_progress = run_manager.update_progress

    def _record_progress(*args: Any, **kwargs: Any) -> Any:
        state = original_update_progress(*args, **kwargs)
        snapshot = state.progress.prediction
        if snapshot is not None:
            progress_snapshots.append(snapshot.model_dump())
        return state

    monkeypatch.setattr(run_manager, "update_progress", _record_progress)

    prediction_executor.execute_prediction(run_state, environment, run_manager)

    assert rf_calls == ["called"]
    assert _stage_sequence(progress_snapshots) == [
        "initializing",
        "resolving_inputs",
        "predicting",
        "rf_filtering",
        "writing_summary",
        "completed",
    ]


def test_prediction_summary_includes_rf_filter_summary(monkeypatch: Any, tmp_path: Path) -> None:
    environment = _build_environment(tmp_path)
    run_manager = RunManager(environment.runs_root)
    run_state = run_manager.create_run(
        "prediction",
        _prediction_spec(apply_rf_filter=True),
    )
    _install_prediction_stubs(monkeypatch, with_rf_filter=False)
    monkeypatch.setattr(
        "alpaca_pipelines.rf.executor.apply_rf_filter",
        lambda **_kwargs: {
            "applied": True,
            "rf_model_path": "/models/rf.joblib",
            "rf_threshold": 0.4,
            "base_detections": 3,
            "rf_passed": 2,
            "rf_rejected": 1,
            "rf_unscored": 0,
            "rejection_rate": 0.333333,
            "pass_rate": 0.666667,
            "files": [],
        },
    )

    prediction_executor.execute_prediction(run_state, environment, run_manager)

    summary = read_json(
        Path(run_state.run_dir) / "outputs" / "predictions" / "prediction_summary.json"
    )
    assert summary["total_detections"] == 3
    assert summary["rf_filtered"] is True
    assert summary["rf_filter_summary"]["rf_passed"] == 2
    assert summary["rf_filter_summary"]["rf_rejected"] == 1
