from __future__ import annotations

import json
import sys
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
