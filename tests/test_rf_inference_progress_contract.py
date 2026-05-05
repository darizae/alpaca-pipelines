from __future__ import annotations

from pathlib import Path
from typing import Any

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.rf_inference import executor as rf_inference_executor
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


def _stage_sequence(progress_snapshots: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []
    for item in progress_snapshots:
        stage = str(item["stage"])
        if not sequence or sequence[-1] != stage:
            sequence.append(stage)
    return sequence


def test_rf_inference_progress_snapshot_contract(monkeypatch: Any, tmp_path: Path) -> None:
    environment = _build_environment(tmp_path)
    run_manager = RunManager(environment.runs_root)

    source_predictions_dir = tmp_path / "source_predictions"
    source_predictions_dir.mkdir()
    write_json(source_predictions_dir / "a.json", {"detections": [{"start_s": 0.0, "end_s": 0.2}]})
    write_json(source_predictions_dir / "b.json", {"detections": [{"start_s": 0.1, "end_s": 0.3}]})
    write_json(
        source_predictions_dir / "prediction_summary.json",
        {
            "total_detections": 2,
            "detection_threshold": 0.8,
            "rf_filtered": False,
            "files": [
                {"audio_file": str(tmp_path / "a.wav"), "n_windows": 10, "n_detections": 1},
                {"audio_file": str(tmp_path / "b.wav"), "n_windows": 10, "n_detections": 1},
            ],
        },
    )

    run_state = run_manager.create_run(
        "rf_inference",
        {
            "source_prediction_run_id": "pred-run-1",
            "rf_training_run_id": "rf-run-1",
            "source_predictions_dir": str(source_predictions_dir),
            "rf_model_path": "/models/rf.joblib",
            "rf_threshold": 0.6,
            "rf_feature_config": {},
            "run_name": "rf-inference-progress-test",
        },
    )

    monkeypatch.setattr(
        rf_inference_executor,
        "apply_rf_filter",
        lambda **_kwargs: {
            "applied": True,
            "rf_model_path": "/models/rf.joblib",
            "rf_threshold": 0.6,
            "base_detections": 2,
            "rf_passed": 2,
            "rf_rejected": 0,
            "rf_unscored": 0,
            "rejection_rate": 0.0,
            "pass_rate": 1.0,
            "files": [
                {"audio_file": str(tmp_path / "a.wav"), "rf_passed": 1},
                {"audio_file": str(tmp_path / "b.wav"), "rf_passed": 1},
            ],
        },
    )

    progress_snapshots: list[dict[str, Any]] = []
    original_update_progress = run_manager.update_progress

    def _record_progress(*args: Any, **kwargs: Any) -> Any:
        state = original_update_progress(*args, **kwargs)
        snapshot = state.progress.prediction
        if snapshot is not None:
            progress_snapshots.append(snapshot.model_dump())
        return state

    monkeypatch.setattr(run_manager, "update_progress", _record_progress)

    completed = rf_inference_executor.execute_rf_inference(run_state, environment, run_manager)

    assert completed.status == "completed"
    persisted = read_json(Path(run_state.run_dir) / "run_state.json")
    prediction_progress = persisted["progress"]["prediction"]
    assert prediction_progress["stage"] == "completed"
    assert prediction_progress["files_total"] == 2
    assert prediction_progress["files_completed"] == 2
    assert prediction_progress["detections_so_far"] == 2
    assert prediction_progress["current_file"] is None
    assert prediction_progress["current_file_windows_total"] is None
    assert prediction_progress["current_file_windows_completed"] is None
    assert isinstance(prediction_progress["updated_at"], str)
    assert _stage_sequence(progress_snapshots) == [
        "initializing",
        "resolving_inputs",
        "rf_filtering",
        "writing_summary",
        "completed",
    ]
