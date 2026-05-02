from __future__ import annotations

from pathlib import Path

from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.postprocessing.executor import export_prediction_run_selection_tables


def _write_prediction_fixture(predictions_dir: Path) -> tuple[Path, Path]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    audio_file = "/audio/a.wav"
    write_json(
        predictions_dir / "prediction_summary.json",
        {
            "run_id": "run-a",
            "model_path": "/models/final.pt",
            "n_files": 1,
            "total_detections": 3,
            "detection_threshold": 0.5,
            "files": [
                {
                    "audio_file": audio_file,
                    "n_windows": 3,
                    "n_detections": 3,
                }
            ],
        },
    )
    base_path = predictions_dir / "a.json"
    rf_path = predictions_dir / "a_rf_filtered.json"
    detections = [
        {"start_s": 0.0, "end_s": 0.1, "score": 0.9, "rf_score": 0.9, "rf_pass": True},
        {"start_s": 0.2, "end_s": 0.3, "score": 0.8, "rf_score": 0.2, "rf_pass": False},
        {"start_s": 0.4, "end_s": 0.5, "score": 0.7, "rf_score": None, "rf_pass": False},
    ]
    write_json(
        base_path,
        {
            "audio_file": audio_file,
            "n_windows": 3,
            "n_detections": 3,
            "detections": detections,
            "scores_shape": [3, 2],
        },
    )
    write_json(
        rf_path,
        {
            "audio_file": audio_file,
            "n_windows": 3,
            "n_detections": 3,
            "detections": detections,
            "scores_shape": [3, 2],
            "rf_filtered": True,
            "rf_model_path": "/models/rf.joblib",
            "rf_threshold": 0.4,
        },
    )
    return base_path, rf_path


def test_selection_table_export_reports_base_exported_detection_count(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    _write_prediction_fixture(predictions_dir)

    selection_tables_dir = tmp_path / "selection_tables_base"
    summary = export_prediction_run_selection_tables(
        predictions_dir=predictions_dir,
        selection_tables_dir=selection_tables_dir,
        use_rf_filtered=False,
    )

    persisted = read_json(selection_tables_dir / "selection_tables_summary.json")
    assert persisted == summary
    assert summary["source_mode"] == "base"
    assert summary["n_exported_detections"] == 3
    assert summary["files"][0]["n_exported_detections"] == 3


def test_selection_table_export_reports_rf_filtered_exported_detection_count(
    tmp_path: Path,
) -> None:
    predictions_dir = tmp_path / "predictions"
    _write_prediction_fixture(predictions_dir)

    selection_tables_dir = tmp_path / "selection_tables_rf"
    summary = export_prediction_run_selection_tables(
        predictions_dir=predictions_dir,
        selection_tables_dir=selection_tables_dir,
        use_rf_filtered=True,
    )

    assert summary["source_mode"] == "rf_filtered"
    assert summary["n_exported_detections"] == 1
    assert summary["files"][0]["n_exported_detections"] == 1
