from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpaca_pipelines.cli import main
from alpaca_pipelines.contracts import RunState, WorkflowOperation


def _run_state() -> RunState:
    return RunState(
        run_id="11111111-1111-1111-1111-111111111111",
        run_type="training",
        status="submitted",
        created_at="2026-03-10T10:00:00Z",
        submitted_at="2026-03-10T10:05:00Z",
        slurm_job_id="12345",
        run_dir="/runs/training/11111111-1111-1111-1111-111111111111",
    )


def test_submit_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(submit_run=lambda run_id, slurm_config=None: _run_state())
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(sys, "argv", ["alpaca-pipelines", "submit", "--run-id", "id", "--json"])

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "submitted"
    assert payload["slurm_job_id"] == "12345"


def test_submit_json_failure_writes_only_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        submit_run=lambda run_id, slurm_config=None: (_ for _ in ()).throw(
            RuntimeError("submission failed")
        )
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(sys, "argv", ["alpaca-pipelines", "submit", "--run-id", "id", "--json"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == "submission failed"


def test_create_json_validation_failure_writes_single_line_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid_prediction.json"
    config_path.write_text("{}", encoding="utf-8")
    api = SimpleNamespace(create_prediction_run=lambda spec: _run_state())
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "create",
            "prediction",
            "--config",
            str(config_path),
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "PredictionRunSpec" in captured.err


def test_fail_operation_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operation = WorkflowOperation(
        job_id="11111111-1111-1111-1111-111111111111",
        workflow="dataset_builder",
        kind="build",
        status="failed",
        created_at="2026-03-12T01:00:00Z",
        started_at="2026-03-12T01:00:01Z",
        finished_at="2026-03-12T01:05:00Z",
        job_dir="/runs/operations/dataset_builder/build/11111111-1111-1111-1111-111111111111",
        error="Marked failed by operator as stale.",
        error_kind="StaleOperation",
    )
    api = SimpleNamespace(
        fail_workflow_operation=lambda job_id, error, error_kind: operation.model_dump()
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "fail-operation",
            "--job-id",
            "id",
            "--error-kind",
            "StaleOperation",
            "--error",
            "Marked failed by operator as stale.",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert payload["error_kind"] == "StaleOperation"


def test_standardizer_import_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operation = WorkflowOperation(
        job_id="22222222-1111-1111-1111-111111111111",
        workflow="standardizer",
        kind="import",
        status="pending",
        created_at="2026-03-12T01:00:00Z",
        started_at="2026-03-12T01:00:01Z",
        finished_at=None,
        job_dir="/runs/operations/standardizer/import/22222222-1111-1111-1111-111111111111",
        metadata={},
    )
    api = SimpleNamespace(
        start_standardizer_import=lambda identity_map_path: operation.model_dump()
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "standardizer-import",
            "--identity-map",
            "/tmp/identity-map.json",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["kind"] == "import"
    assert payload["workflow"] == "standardizer"


def test_dataset_apply_review_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        start_apply_review=lambda dataset_name, target_review_table_path, noise_review_table_path: {
            "job_id": "33333333-1111-1111-1111-111111111111",
            "workflow": "dataset_builder",
            "kind": "apply_review",
            "status": "pending",
            "created_at": "2026-03-12T01:00:00Z",
            "started_at": "2026-03-12T01:00:01Z",
            "finished_at": None,
            "job_dir": (
                "/runs/operations/dataset_builder/apply_review/33333333-1111-1111-1111-111111111111"
            ),
            "metadata": {"dataset_name": dataset_name},
            "spec": {
                "target_review_table_path": target_review_table_path,
                "noise_review_table_path": noise_review_table_path,
            },
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "dataset-apply-review",
            "--dataset-name",
            "dataset-a",
            "--target-review-table",
            "/tmp/target.txt",
            "--noise-review-table",
            "/tmp/noise.txt",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["kind"] == "apply_review"
    assert payload["metadata"]["dataset_name"] == "dataset-a"
    assert payload["spec"]["target_review_table_path"] == "/tmp/target.txt"
    assert payload["spec"]["noise_review_table_path"] == "/tmp/noise.txt"


def test_delete_failed_operation_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        delete_failed_workflow_operation=lambda job_id: {
            "job_id": job_id,
            "workflow": "dataset_builder",
            "kind": "build",
            "status": "failed",
            "job_dir": "/runs/operations/dataset_builder/build/id",
            "deleted": True,
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "delete-failed-operation",
            "--job-id",
            "id",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["job_id"] == "id"
    assert payload["deleted"] is True


def test_prediction_review_preview_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        generate_prediction_review_preview=lambda manifest_path, item_id, spectrogram_config: {
            "mode": "preview",
            "prediction_run_id": "run-1",
            "session_id": "session-1",
            "item_id": item_id,
            "summary_path": (
                "/runs/prediction/run-1/outputs/manual_review/session-1/preview_item-1.json"
            ),
            "item": {},
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "prediction-review-preview",
            "--manifest",
            "/tmp/session.json",
            "--item-id",
            "item-1",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["mode"] == "preview"
    assert payload["item_id"] == "item-1"


def test_prediction_review_generate_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        generate_prediction_review_batch=lambda manifest_path, spectrogram_config: {
            "mode": "batch",
            "prediction_run_id": "run-1",
            "session_id": "session-1",
            "summary_path": "/runs/prediction/run-1/outputs/manual_review/session-1/summary.json",
            "n_items": 2,
            "items": [],
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "prediction-review-generate",
            "--manifest",
            "/tmp/session.json",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["mode"] == "batch"
    assert payload["n_items"] == 2


def test_prediction_review_concat_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        concatenate_prediction_review_clips=lambda manifest_path, output_wav: {
            "prediction_run_id": "run-1",
            "session_id": "session-1",
            "concat_wav": (
                "/runs/prediction/run-1/outputs/manual_review/session-1/raven/review_concat.wav"
            ),
            "n_items": 2,
            "items": [],
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "prediction-review-concat",
            "--manifest",
            "/tmp/session.json",
            "--output-wav",
            "/tmp/review_concat.wav",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["n_items"] == 2
    assert payload["concat_wav"].endswith("review_concat.wav")


def test_prediction_review_export_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        export_prediction_review_artifacts=lambda manifest_path, destination_dir, item_id: {
            "prediction_run_id": "run-1",
            "session_id": "session-1",
            "destination_dir": str(destination_dir / "session-1"),
            "summary_path": str(destination_dir / "session-1" / "export_summary.json"),
            "n_items": 1,
            "items": [],
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "prediction-review-export",
            "--manifest",
            "/tmp/session.json",
            "--destination-dir",
            "/tmp/export",
            "--item-id",
            "item-1",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["n_items"] == 1


def test_prediction_review_materialize_curated_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        materialize_curated_prediction_examples=lambda **_: {
            "curated_source_root": "/datasets/_curated_prediction_examples",
            "manifest_paths": [
                "/datasets/_curated_prediction_examples/audio_collection_a/run-1/session-1/manifest.json"
            ],
            "prediction_run_id": "run-1",
            "review_session_id": "session-1",
            "counts_by_label": {"target": 2, "noise": 1},
            "created_count": 3,
            "updated_count": 0,
            "skipped_count": 0,
            "total_items": 3,
            "source_recording_keys": ["401_20250211_075558"],
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "prediction-review-materialize-curated",
            "--manifest",
            "/tmp/session.json",
            "--labels",
            "/tmp/labels.json",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["prediction_run_id"] == "run-1"
    assert payload["counts_by_label"]["target"] == 2


def test_prediction_review_materialize_curated_accepts_curated_export_manifest_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called: dict[str, object] = {}

    def _materialize_curated_prediction_examples(**kwargs: object) -> dict[str, object]:
        called.update(kwargs)
        return {
            "curated_source_root": "/datasets/_curated_prediction_examples",
            "manifest_paths": [],
            "prediction_run_id": "run-1",
            "review_session_id": "session-1",
            "counts_by_label": {"target": 0, "noise": 0},
            "created_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "total_items": 0,
            "source_recording_keys": [],
        }

    api = SimpleNamespace(
        materialize_curated_prediction_examples=_materialize_curated_prediction_examples
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "prediction-review-materialize-curated",
            "--curated-export-manifest",
            "/tmp/curated-export.json",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["prediction_run_id"] == "run-1"
    assert called["manifest_path"] is None
    assert called["labels_path"] is None
    assert called["curated_export_manifest"] == Path("/tmp/curated-export.json")


def test_curated_source_status_json_writes_single_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = SimpleNamespace(
        list_curated_prediction_sources=lambda destination_root: {
            "curated_source_root": "/datasets/_curated_prediction_examples",
            "manifests": [],
            "counts_by_collection": {},
            "counts_by_label": {},
            "counts_by_provenance_type": {},
            "warnings": [],
        }
    )
    monkeypatch.setattr("alpaca_pipelines.cli._get_api", lambda: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca-pipelines",
            "curated-source-status",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["curated_source_root"] == "/datasets/_curated_prediction_examples"
