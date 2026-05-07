from __future__ import annotations

from pathlib import Path

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.runs.manager import RunManager
from alpaca_pipelines.runs.review_index import build_review_index_summary
from alpaca_pipelines.runs.review_index_backfill import backfill_review_index_summaries


def _build_environment(tmp_path: Path) -> PipelineEnvironment:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir(parents=True)
    datasets_root.mkdir(parents=True)
    write_json(collection_root / "merged_index.json", {"meta": {}, "entries": []})
    return PipelineEnvironment.from_explicit(
        collection_root=collection_root,
        merged_index_path=collection_root / "merged_index.json",
        datasets_root=datasets_root,
        runs_root=runs_root,
    )


def test_build_review_index_summary_includes_rf_partitions() -> None:
    predictions_dir = Path("/tmp/runs/prediction/run-1/outputs/predictions")
    payload = build_review_index_summary(
        run_id="run-1",
        run_type="prediction",
        predictions_dir=predictions_dir,
        prediction_summary={
            "rf_filtered": True,
            "total_detections": 5,
            "files": [
                {"audio_file": "/audio/a.wav", "n_windows": 10, "n_detections": 3},
                {"audio_file": "/audio/b.wav", "n_windows": 8, "n_detections": 2},
            ],
            "rf_filter_summary": {
                "rf_passed": 2,
                "rf_rejected": 2,
                "rf_unscored": 1,
                "files": [
                    {
                        "audio_file": "/audio/a.wav",
                        "rf_passed": 1,
                        "rf_rejected": 1,
                        "rf_unscored": 1,
                    }
                ],
            },
        },
    )

    assert payload["run_id"] == "run-1"
    assert payload["n_files"] == 2
    assert payload["rf_partition_totals"] == {"accepted": 2, "rejected": 2, "unscored": 1}
    assert payload["files"][0]["rf_partitions"] == {"accepted": 1, "rejected": 1, "unscored": 1}


def test_backfill_review_index_summaries_writes_artifact_for_completed_prediction_run(
    tmp_path: Path,
) -> None:
    environment = _build_environment(tmp_path)
    run_manager = RunManager(environment.runs_root)
    run_state = run_manager.create_run("prediction", {"model_path": "/tmp/model.pt"})
    run_manager.mark_running(run_state.run_id)
    completed = run_manager.mark_completed(run_state.run_id)

    predictions_dir = Path(completed.outputs.predictions_dir or "")
    write_json(
        predictions_dir / "prediction_summary.json",
        {
            "run_id": completed.run_id,
            "rf_filtered": False,
            "total_detections": 4,
            "files": [
                {
                    "audio_file": "/tmp/audio.wav",
                    "n_windows": 11,
                    "n_detections": 4,
                }
            ],
        },
    )

    summary = backfill_review_index_summaries(run_manager)
    assert summary.scanned == 1
    assert summary.migrated == [completed.run_id]
    assert not summary.failed

    refreshed = run_manager.find_run(completed.run_id)
    review_index_path = Path(refreshed.outputs.prediction_review_index_summary_path or "")
    assert review_index_path.is_file()
    payload = read_json(review_index_path)
    assert payload["run_id"] == completed.run_id
    assert payload["n_files"] == 1
