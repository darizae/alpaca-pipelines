from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from alpaca_pipelines.api import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.training.config import TrainingRunSpec


def _build_api(tmp_path: Path) -> PipelineAPI:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir()
    datasets_root.mkdir()
    (datasets_root / "dataset-a").mkdir()
    write_json(collection_root / "merged_index.json", {"meta": {}, "entries": []})
    environment = PipelineEnvironment.from_explicit(
        collection_root=collection_root,
        merged_index_path=collection_root / "merged_index.json",
        datasets_root=datasets_root,
        runs_root=runs_root,
    )
    return PipelineAPI(environment)


def _training_spec() -> TrainingRunSpec:
    return TrainingRunSpec(dataset_name="dataset-a")


def test_submit_run_persists_submitted_at_and_slurm_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    script_path = Path(run_state.run_dir) / "slurm" / "job.sbatch"

    monkeypatch.setattr("alpaca_pipelines.api.generate_slurm_script", lambda **_: script_path)
    monkeypatch.setattr(
        "alpaca_pipelines.api.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="12345\n",
            stderr="",
        ),
    )

    submitted = api.submit_run(run_state.run_id)

    persisted = read_json(Path(run_state.run_dir) / "run_state.json")
    assert submitted.status == "submitted"
    assert submitted.slurm_job_id == "12345"
    assert submitted.submitted_at is not None
    assert persisted["slurm_job_id"] == "12345"
    assert persisted["submitted_at"] == submitted.submitted_at


def test_submit_run_failed_sbatch_leaves_state_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    script_path = Path(run_state.run_dir) / "slurm" / "job.sbatch"

    monkeypatch.setattr("alpaca_pipelines.api.generate_slurm_script", lambda **_: script_path)
    monkeypatch.setattr(
        "alpaca_pipelines.api.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="submission error",
        ),
    )

    with pytest.raises(RuntimeError, match="submission error"):
        api.submit_run(run_state.run_id)

    persisted = read_json(Path(run_state.run_dir) / "run_state.json")
    assert persisted["status"] == "created"
    assert persisted["submitted_at"] is None
    assert persisted["slurm_job_id"] is None


def test_cancel_created_run_marks_cancelled(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())

    cancelled = api.cancel_run(run_state.run_id)

    assert cancelled.status == "cancelled"
    assert cancelled.completed_at is not None


def test_cancel_submitted_run_calls_scancel_and_marks_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    api.run_manager.mark_submitted(run_state.run_id, "98765")
    calls: list[list[str]] = []

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        command = args[0]
        assert isinstance(command, list)
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("alpaca_pipelines.api.subprocess.run", _fake_run)

    cancelled = api.cancel_run(run_state.run_id)

    assert calls == [["scancel", "98765"]]
    assert cancelled.status == "cancelled"
    assert cancelled.completed_at is not None


def test_cancel_submitted_run_without_slurm_job_id_fails(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    persisted = read_json(Path(run_state.run_dir) / "run_state.json")
    persisted["status"] = "submitted"
    persisted["submitted_at"] = "2026-03-10T12:00:00Z"
    write_json(Path(run_state.run_dir) / "run_state.json", persisted)

    with pytest.raises(ValueError, match="without slurm_job_id"):
        api.cancel_run(run_state.run_id)


def test_migrate_backend_meta_backfills_missing_submission_fields(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    run_dir = Path(run_state.run_dir)
    write_json(
        run_dir / "backend_meta.json",
        {
            "submitted_at": "2026-03-10T12:00:00Z",
            "slurm_job_id": "12345",
        },
    )

    summary = api.migrate_backend_meta()

    persisted = read_json(run_dir / "run_state.json")
    assert summary.migrated == [str(run_dir)]
    assert persisted["submitted_at"] == "2026-03-10T12:00:00Z"
    assert persisted["slurm_job_id"] == "12345"


def test_migrate_backend_meta_is_idempotent(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    run_dir = Path(run_state.run_dir)
    write_json(
        run_dir / "backend_meta.json",
        {
            "submitted_at": "2026-03-10T12:00:00Z",
            "slurm_job_id": "12345",
        },
    )

    first = api.migrate_backend_meta()
    second = api.migrate_backend_meta()

    persisted = read_json(run_dir / "run_state.json")
    assert first.migrated == [str(run_dir)]
    assert second.migrated == []
    assert persisted["submitted_at"] == "2026-03-10T12:00:00Z"
    assert persisted["slurm_job_id"] == "12345"


def test_migrate_backend_meta_ignores_alpaca_ui_jobs(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    ui_job_dir = api.environment.runs_root / "alpaca-ui-jobs" / "job-1"
    ui_job_dir.mkdir(parents=True)
    write_json(ui_job_dir / "state.json", {"status": "pending"})

    summary = api.migrate_backend_meta()

    assert all("alpaca-ui-jobs" not in path for path in summary.migrated)
    assert all("alpaca-ui-jobs" not in path for path in summary.skipped)
    assert all("alpaca-ui-jobs" not in path for path in summary.inconsistent)


def test_migrate_backend_meta_fails_on_conflicting_existing_values(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    run_state = api.create_training_run(_training_spec())
    run_dir = Path(run_state.run_dir)
    persisted = read_json(run_dir / "run_state.json")
    persisted["submitted_at"] = "2026-03-10T12:00:00Z"
    persisted["slurm_job_id"] = "99999"
    write_json(run_dir / "run_state.json", persisted)
    write_json(
        run_dir / "backend_meta.json",
        {
            "submitted_at": "2026-03-10T12:00:00Z",
            "slurm_job_id": "12345",
        },
    )

    with pytest.raises(ValueError, match="Inconsistent backend_meta migration state"):
        api.migrate_backend_meta()


def test_migrate_backend_meta_fails_when_sidecar_exists_without_run_state(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    orphan_dir = api.environment.runs_root / "training" / "orphan-run"
    orphan_dir.mkdir(parents=True)
    write_json(
        orphan_dir / "backend_meta.json",
        {
            "submitted_at": "2026-03-10T12:00:00Z",
            "slurm_job_id": "12345",
        },
    )

    with pytest.raises(ValueError, match="Inconsistent backend_meta migration state"):
        api.migrate_backend_meta()


def test_fail_workflow_operation_marks_pending_job_failed(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    operation = api.start_dataset_build(
        strategy_name="dataset-a",
        strategy_config={
            "target_collection_names": ["audio_collection_alpha"],
            "noise_collection_names": ["audio_collection_alpha"],
            "split_strategy": "clipwise_balanced",
            "seed": 42,
            "min_quality": 2,
            "noise_per_positive": 1.0,
            "noise_mining": {
                "attempts_per_slot": 20,
                "source_category_dirs": ["clips_labelled"],
                "low_quality_as_negative": True,
                "low_quality_threshold": 1,
            },
            "split_fractions": [0.7, 0.15, 0.15],
            "duration_tolerance_s": 0.1,
            "review_gap_s": 0.5,
            "freq_low_hz": 0,
            "freq_high_hz": 4000,
        },
    )

    failed = api.fail_workflow_operation(
        job_id=operation["job_id"],
        error="Marked failed by operator as stale.",
        error_kind="StaleOperation",
    )

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert failed["status"] == "failed"
    assert failed["error_kind"] == "StaleOperation"
    assert failed["finished_at"] is not None
    assert persisted["status"] == "failed"


def test_delete_failed_workflow_operation_removes_job_dir(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    operation = api.start_dataset_build(
        strategy_name="dataset-a",
        strategy_config={
            "target_collection_names": ["audio_collection_alpha"],
            "noise_collection_names": ["audio_collection_alpha"],
            "split_strategy": "clipwise_balanced",
            "seed": 42,
            "min_quality": 2,
            "noise_per_positive": 1.0,
            "noise_mining": {
                "attempts_per_slot": 20,
                "source_category_dirs": ["clips_labelled"],
                "low_quality_as_negative": True,
                "low_quality_threshold": 1,
            },
            "split_fractions": [0.7, 0.15, 0.15],
            "duration_tolerance_s": 0.1,
            "review_gap_s": 0.5,
            "freq_low_hz": 0,
            "freq_high_hz": 4000,
        },
    )
    api.fail_workflow_operation(
        job_id=operation["job_id"],
        error="Marked failed by operator as stale.",
        error_kind="StaleOperation",
    )

    deleted = api.delete_failed_workflow_operation(job_id=operation["job_id"])

    assert deleted["job_id"] == operation["job_id"]
    assert deleted["deleted"] is True
    assert not Path(operation["job_dir"]).exists()
