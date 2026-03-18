from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from alpaca_pipelines.api import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.recordings import SourceRecording
from alpaca_pipelines.workflow_ops import _plan_hash


def _build_api(tmp_path: Path) -> PipelineAPI:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir()
    datasets_root.mkdir()
    runs_root.mkdir()
    write_json(
        collection_root / "merged_index.json",
        {
            "meta": {
                "generated_at": "2026-03-13T00:00:00Z",
                "n_collections": 0,
                "n_total_hums": 0,
            },
            "entries": [],
        },
    )
    environment = PipelineEnvironment.from_explicit(
        collection_root=collection_root,
        merged_index_path=collection_root / "merged_index.json",
        datasets_root=datasets_root,
        runs_root=runs_root,
    )
    return PipelineAPI(environment)


def _disable_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.WorkflowOperationManager._spawn_worker",
        lambda self, paths: None,
    )


def test_spawn_worker_uses_cli_module_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    calls: list[dict[str, object]] = []

    def _fake_popen(
        args: list[str],
        *,
        start_new_session: bool,
        stdout: object,
        stderr: object,
    ) -> SimpleNamespace:
        calls.append(
            {
                "args": args,
                "start_new_session": start_new_session,
                "stdout_name": getattr(stdout, "name", ""),
                "stderr_name": getattr(stderr, "name", ""),
            }
        )
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr("alpaca_pipelines.workflow_ops.subprocess.Popen", _fake_popen)

    operation = api.start_dataset_build(
        strategy_name="dataset-a",
        strategy_config={
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

    assert len(calls) == 1
    args = calls[0]["args"]
    assert isinstance(args, list)
    assert args[:4] == [sys.executable, "-m", "alpaca_pipelines.cli", "_execute-operation"]
    assert calls[0]["start_new_session"] is True
    assert str(calls[0]["stdout_name"]).endswith("stdout.log")
    assert str(calls[0]["stderr_name"]).endswith("stderr.log")
    assert operation["metadata"]["launch_mode"] == "detached_worker"
    assert operation["metadata"]["worker_pid"] == 1234


def test_python_module_invocation_executes_operation_worker(tmp_path: Path) -> None:
    api = _build_api(tmp_path)
    collection_dir = api.environment.collection_root / "audio_collection_alpha"
    (collection_dir / "clips_labelled").mkdir(parents=True)
    (collection_dir / "hums_segmented").mkdir(parents=True)
    (collection_dir / "clips_labelled" / "clip.wav").write_bytes(b"")
    (collection_dir / "hums_segmented" / "hum.wav").write_bytes(b"")

    job_dir = api.environment.runs_root / "operations" / "standardizer" / "scan" / "job-1"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "operation.json",
        {
            "job_id": "job-1",
            "workflow": "standardizer",
            "kind": "scan",
            "status": "pending",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": None,
            "job_dir": str(job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {},
        },
    )
    write_json(job_dir / "spec.json", {"workflow": "standardizer", "kind": "scan", "spec": {}})

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo_root / "src"),
            "ALPACA_COLLECTION_ROOT": str(api.environment.collection_root),
            "ALPACA_MERGED_INDEX": str(api.environment.merged_index_path),
            "ALPACA_DATASETS_ROOT": str(api.environment.datasets_root),
            "ALPACA_RUNS_ROOT": str(api.environment.runs_root),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpaca_pipelines.cli",
            "_execute-operation",
            "--job-dir",
            str(job_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    operation = read_json(job_dir / "operation.json")
    assert operation["status"] == "completed"
    assert operation["result_summary"]["collections"] == 1
    assert operation["result_summary"]["total_clips"] == 1
    assert operation["result_summary"]["total_hums"] == 1


def test_standardizer_scan_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    collection_dir = api.environment.collection_root / "audio_collection_alpha"
    collection_dir.mkdir()

    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.scan_root",
        lambda root: SimpleNamespace(
            root=root,
            payload={
                "collections": [
                    {
                        "collection": collection_dir.name,
                        "clips_dir": str(collection_dir / "clips_labelled"),
                        "hums_dir": str(collection_dir / "hums_segmented"),
                        "n_clips": 2,
                        "n_hums": 3,
                    }
                ]
            },
        ),
    )

    operation = api.start_standardizer_scan()
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["collections"] == 1
    assert persisted["result_summary"]["total_clips"] == 2
    assert persisted["result_summary"]["total_hums"] == 3


def test_standardizer_import_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    identity_map_path = tmp_path / "identity_map.json"
    write_json(
        identity_map_path,
        {
            "canonical": {"401": {"display_name": "401"}},
            "aliases": {"401": "401"},
        },
    )

    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.import_raw_batches_from_identity_map_path",
        lambda **kwargs: SimpleNamespace(
            imported_batches=["401_m28_20250213"],
            imported_collection_dirs=[
                str(api.environment.collection_root / "audio_collection_401_m28_20250213")
            ],
            imported_recordings=[
                SourceRecording(
                    key="401_20250211_075558",
                    collection="audio_collection_401_m28_20250213",
                    subject_id="401",
                    wav_path="audio_collection_401_m28_20250213/raw_recordings/20250211_075558.WAV",
                )
            ],
            matched_csv_count=1,
            missing_csv_count=0,
        ),
    )

    operation = api.start_standardizer_import(str(identity_map_path))
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["imported_batches"] == 1
    assert persisted["result_summary"]["imported_recordings"] == 1
    assert persisted["result_summary"]["matched_csv_count"] == 1
    assert persisted["result_summary"]["batch_names"] == ["401_m28_20250213"]


def test_standardizer_plan_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    collection_dir = api.environment.collection_root / "audio_collection_alpha"
    collection_dir.mkdir()
    identity_map_path = tmp_path / "identity_map.json"
    write_json(
        identity_map_path,
        {
            "canonical": {"alpha": {"display_name": "Alpha"}},
            "aliases": {"alpha": "alpha"},
        },
    )

    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.find_collection_dirs",
        lambda root, fs: [collection_dir],
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.plan_renames_for_collection",
        lambda **kwargs: ([], [], [], [], []),
    )

    operation = api.start_standardizer_plan(str(identity_map_path))
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["plan_id"] == operation["job_id"]
    assert Path(Path(operation["job_dir"]) / "plan.json").is_file()


def test_standardizer_plan_operation_includes_non_ready_collections_with_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    ready_collection_dir = api.environment.collection_root / "audio_collection_ready"
    raw_only_collection_dir = api.environment.collection_root / "audio_collection_raw_only"
    ready_collection_dir.mkdir()
    (ready_collection_dir / "clips_labelled").mkdir()
    (ready_collection_dir / "hums_segmented").mkdir()
    raw_only_collection_dir.mkdir()
    (raw_only_collection_dir / "raw_recordings").mkdir()
    identity_map_path = tmp_path / "identity_map.json"
    write_json(
        identity_map_path,
        {
            "canonical": {"alpha": {"display_name": "Alpha"}},
            "aliases": {"alpha": "alpha"},
        },
    )

    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.find_collection_dirs",
        lambda root, fs: [ready_collection_dir, raw_only_collection_dir],
    )

    def _fake_plan_renames_for_collection(
        **kwargs: object,
    ) -> tuple[list[object], list[object], list[object], list[object], list[object]]:
        collection_dir = kwargs["collection_dir"]
        if collection_dir == raw_only_collection_dir:
            return [], [], [], [], []
        return [], [], [], [], []

    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.plan_renames_for_collection",
        _fake_plan_renames_for_collection,
    )

    operation = api.start_standardizer_plan(str(identity_map_path))
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["collections"] == [
        {
            "name": "audio_collection_ready",
            "status": "ready",
            "has_clips": True,
            "has_hums": True,
            "has_raw_recordings": False,
            "clip_renames": [],
            "raw_recording_renames": [],
            "dir_renames": [],
        },
        {
            "name": "audio_collection_raw_only",
            "status": "raw_only",
            "has_clips": False,
            "has_hums": False,
            "has_raw_recordings": True,
            "clip_renames": [],
            "raw_recording_renames": [],
            "dir_renames": [],
        },
    ]


def test_standardizer_apply_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    plan_job_dir = api.environment.runs_root / "operations" / "standardizer" / "plan" / "plan-1"
    plan_job_dir.mkdir(parents=True)
    plan_path = plan_job_dir / "plan.json"
    write_json(plan_path, {"root": str(api.environment.collection_root), "ops": [], "audit": {}})
    write_json(
        plan_job_dir / "operation.json",
        {
            "job_id": "plan-1",
            "workflow": "standardizer",
            "kind": "plan",
            "status": "completed",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": "2026-03-13T00:01:00Z",
            "job_dir": str(plan_job_dir),
            "artifact_path": str(plan_path),
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {},
        },
    )
    monkeypatch.setattr("alpaca_pipelines.workflow_ops.apply_rename_plan_file", lambda path: 4)

    operation = api.start_standardizer_apply(
        plan_job_id="plan-1",
        confirmation_phrase="APPLY {}".format(_plan_hash(plan_path)),
    )
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["ops_applied"] == 4


def test_standardizer_index_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    identity_map_path = tmp_path / "identity_map.json"
    write_json(
        identity_map_path,
        {
            "canonical": {"alpha": {"display_name": "Alpha"}},
            "aliases": {"alpha": "alpha"},
        },
    )

    def _fake_build_indexes_from_identity_map_path(
        *,
        root: Path,
        identity_map_path: Path,
        out_dir: Path,
        min_source_quality_to_keep: int,
    ) -> SimpleNamespace:
        del root, identity_map_path, min_source_quality_to_keep
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / "merged_index.json",
            {
                "meta": {"generated_at": "2026-03-13T00:00:00Z"},
                "entries": [{"keep": True}, {"keep": False}],
            },
        )
        return SimpleNamespace(
            merged_payload={
                "meta": {"generated_at": "2026-03-13T00:00:00Z"},
                "entries": [{"keep": True}],
            },
            per_collection_payloads={"audio_collection_alpha": {"entries": [{"keep": False}]}},
        )

    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.build_indexes_from_identity_map_path",
        _fake_build_indexes_from_identity_map_path,
    )

    operation = api.start_standardizer_index(
        identity_map_path=str(identity_map_path),
        min_quality=1,
    )
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["kept"] == 1
    assert persisted["result_summary"]["excluded"] == 1


def test_dataset_build_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    dataset_dir = api.environment.datasets_root / "strategy-a"
    dataset_dir.mkdir()
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.build_dataset",
        lambda **kwargs: SimpleNamespace(
            dataset_dir=dataset_dir,
            n_target=5,
            n_noise=6,
            splits={"train": 7, "val": 2, "test": 2},
        ),
    )

    operation = api.start_dataset_build(
        strategy_name="strategy-a",
        strategy_config={
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
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["dataset_dir"] == str(dataset_dir)
    assert persisted["result_summary"]["n_target"] == 5


def test_dataset_prepare_review_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    dataset_dir = api.environment.datasets_root / "dataset-a"
    dataset_dir.mkdir()
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.prepare_review",
        lambda path: SimpleNamespace(
            concat_wav_path=path / "review" / "review.wav",
            selection_table_path=path / "review" / "review.Table.1.selections.txt",
        ),
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.load_manifest",
        lambda path: SimpleNamespace(meta=SimpleNamespace(n_snippets=12)),
    )

    operation = api.start_prepare_review("dataset-a")
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["dataset_name"] == "dataset-a"
    assert persisted["result_summary"]["n_snippets"] == 12


def test_dataset_apply_review_operation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    dataset_dir = api.environment.datasets_root / "dataset-a"
    dataset_dir.mkdir()
    review_table_path = tmp_path / "review.txt"
    review_table_path.write_text("uid\tSound_type\n", encoding="utf-8")
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.apply_review",
        lambda dataset_dir, review_table_path: SimpleNamespace(
            n_corrections=3,
            n_discarded=1,
            updated_manifest=SimpleNamespace(meta=SimpleNamespace(n_target=10, n_noise=8)),
        ),
    )

    operation = api.start_apply_review("dataset-a", str(review_table_path))
    api.execute_workflow_operation(operation["job_dir"])

    persisted = read_json(Path(operation["job_dir"]) / "operation.json")
    assert persisted["status"] == "completed"
    assert persisted["result_summary"]["n_corrections"] == 3
    assert persisted["result_summary"]["n_target"] == 10


def test_dataset_status_marks_dead_detached_job_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    job_dir = api.environment.runs_root / "operations" / "dataset_builder" / "build" / "job-1"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "operation.json",
        {
            "job_id": "job-1",
            "workflow": "dataset_builder",
            "kind": "build",
            "status": "pending",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": None,
            "job_dir": str(job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {
                "strategy_name": "strategy-a",
                "launch_mode": "detached_worker",
                "worker_pid": 999999,
            },
        },
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.WorkflowOperationManager._worker_process_is_active",
        staticmethod(lambda worker_pid, job_dir: False),
    )

    payload = api.get_dataset_builder_status()

    assert payload["active_jobs"] == []
    operation = api.get_workflow_operation("job-1")
    assert operation["status"] == "failed"
    assert operation["error_kind"] == "StaleOperation"


def test_standardizer_status_marks_pidless_detached_job_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    job_dir = api.environment.runs_root / "operations" / "standardizer" / "scan" / "job-1"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "operation.json",
        {
            "job_id": "job-1",
            "workflow": "standardizer",
            "kind": "scan",
            "status": "running",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": None,
            "job_dir": str(job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {
                "launch_mode": "detached_worker",
            },
        },
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.WorkflowOperationManager._worker_process_is_active",
        staticmethod(lambda worker_pid, job_dir: True),
    )

    payload = api.get_standardizer_status()

    assert payload["active_jobs"] == []
    operation = api.get_workflow_operation("job-1")
    assert operation["status"] == "failed"
    assert operation["error"] == "Detached workflow worker metadata is missing worker_pid."


def test_standardizer_status_returns_last_import(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    job_dir = api.environment.runs_root / "operations" / "standardizer" / "import" / "job-1"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "operation.json",
        {
            "job_id": "job-1",
            "workflow": "standardizer",
            "kind": "import",
            "status": "completed",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:01Z",
            "finished_at": "2026-03-13T00:00:02Z",
            "job_dir": str(job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": {"imported_batches": 1},
            "error": None,
            "error_kind": None,
            "metadata": {},
        },
    )

    payload = api.get_standardizer_status()

    assert payload["last_import"]["job_id"] == "job-1"
    assert payload["last_import"]["kind"] == "import"


def test_dataset_status_marks_legacy_active_job_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    job_dir = api.environment.runs_root / "operations" / "dataset_builder" / "build" / "job-1"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "operation.json",
        {
            "job_id": "job-1",
            "workflow": "dataset_builder",
            "kind": "build",
            "status": "pending",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": None,
            "job_dir": str(job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {
                "strategy_name": "strategy-a",
            },
        },
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.WorkflowOperationManager._worker_process_is_active",
        staticmethod(lambda worker_pid, job_dir: True),
    )

    payload = api.get_dataset_builder_status()

    assert payload["active_jobs"] == []
    operation = api.get_workflow_operation("job-1")
    assert operation["status"] == "failed"
    assert operation["error"] == "Workflow operation is missing detached worker launch metadata."


def test_start_dataset_build_ignores_stale_dead_job_for_same_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    _disable_spawn(monkeypatch)
    stale_job_dir = (
        api.environment.runs_root / "operations" / "dataset_builder" / "build" / "stale-job"
    )
    stale_job_dir.mkdir(parents=True)
    write_json(
        stale_job_dir / "operation.json",
        {
            "job_id": "stale-job",
            "workflow": "dataset_builder",
            "kind": "build",
            "status": "pending",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": None,
            "job_dir": str(stale_job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {
                "strategy_name": "strategy-a",
                "launch_mode": "detached_worker",
                "worker_pid": 999999,
            },
        },
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.WorkflowOperationManager._worker_process_is_active",
        staticmethod(lambda worker_pid, job_dir: False),
    )

    operation = api.start_dataset_build(
        strategy_name="strategy-a",
        strategy_config={
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

    stale_operation = api.get_workflow_operation("stale-job")
    assert stale_operation["status"] == "failed"
    assert operation["job_id"] != "stale-job"


def test_dataset_status_keeps_live_detached_job_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    job_dir = api.environment.runs_root / "operations" / "dataset_builder" / "build" / "job-1"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "operation.json",
        {
            "job_id": "job-1",
            "workflow": "dataset_builder",
            "kind": "build",
            "status": "running",
            "created_at": "2026-03-13T00:00:00Z",
            "started_at": "2026-03-13T00:00:00Z",
            "finished_at": None,
            "job_dir": str(job_dir),
            "artifact_path": None,
            "rollback_artifact_path": None,
            "result_summary": None,
            "error": None,
            "error_kind": None,
            "metadata": {
                "strategy_name": "strategy-a",
                "launch_mode": "detached_worker",
                "worker_pid": 1234,
            },
        },
    )
    monkeypatch.setattr(
        "alpaca_pipelines.workflow_ops.WorkflowOperationManager._worker_process_is_active",
        staticmethod(lambda worker_pid, job_dir: True),
    )

    payload = api.get_dataset_builder_status()

    assert len(payload["active_jobs"]) == 1
    assert payload["active_jobs"][0]["job_id"] == "job-1"
