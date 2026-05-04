from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca_pipelines.collections.contracts import load_identity_map
from alpaca_pipelines.collections.fs import FileSystem, RollbackIncompleteError
from alpaca_pipelines.collections.paths import CategoryNames, find_collection_dirs
from alpaca_pipelines.collections.planning.rename_plan import plan_renames_for_collection
from alpaca_pipelines.collections.scanning import scan_collection
from alpaca_pipelines.collections.workflows import (
    apply_rename_plan_file,
    build_indexes_from_identity_map_path,
    import_raw_batches_from_identity_map_path,
    scan_root,
    write_scan_report,
)
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import WorkflowOperation
from alpaca_pipelines.datasets.config import StrategyConfig
from alpaca_pipelines.datasets.workflows import (
    ReviewApplyResult,
    ReviewPrepResult,
    apply_review,
    build_dataset,
    prepare_review,
)
from alpaca_pipelines.io_utils import read_json, write_json

_OPERATION_STATE = "operation.json"
_OPERATION_SPEC = "spec.json"
_RUNNER_STDOUT = "stdout.log"
_RUNNER_STDERR = "stderr.log"
_WORKER_PID_METADATA_KEY = "worker_pid"
_LAUNCH_MODE_METADATA_KEY = "launch_mode"
_DETACHED_WORKER_LAUNCH_MODE = "detached_worker"


def _validate_dataset_build_summary(summary: dict[str, Any], manifest: Any) -> None:
    n_target = int(summary["n_target"])
    n_noise = int(summary["n_noise"])
    n_snippets = int(summary["n_snippets"])
    splits = summary["splits"]

    if n_snippets != n_target + n_noise:
        raise ValueError("Dataset build summary mismatch: n_snippets != n_target + n_noise")
    if sum(int(value) for value in splits.values()) != n_snippets:
        raise ValueError("Dataset build summary mismatch: split totals != n_snippets")
    if manifest.meta.n_target != n_target:
        raise ValueError("Dataset build summary mismatch: manifest n_target differs")
    if manifest.meta.n_noise != n_noise:
        raise ValueError("Dataset build summary mismatch: manifest n_noise differs")
    if manifest.meta.n_snippets != n_snippets:
        raise ValueError("Dataset build summary mismatch: manifest n_snippets differs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class OperationPaths:
    job_dir: Path
    state_path: Path
    spec_path: Path
    stdout_path: Path
    stderr_path: Path


class WorkflowOperationManager:
    def __init__(self, environment: PipelineEnvironment) -> None:
        self.environment = environment
        self.root = environment.runs_root / "operations"

    def start(
        self,
        *,
        workflow: str,
        kind: str,
        spec: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        artifact_name: str | None = None,
        rollback_artifact_name: str | None = None,
    ) -> WorkflowOperation:
        job_id = str(uuid.uuid4())
        paths = self._paths(workflow, kind, job_id)
        paths.job_dir.mkdir(parents=True, exist_ok=False)
        artifact_path = str(paths.job_dir / artifact_name) if artifact_name else None
        rollback_path = (
            str(paths.job_dir / rollback_artifact_name) if rollback_artifact_name else None
        )
        operation = WorkflowOperation(
            job_id=job_id,
            workflow=workflow,
            kind=kind,
            status="pending",
            created_at=_now_iso(),
            started_at=_now_iso(),
            job_dir=str(paths.job_dir),
            artifact_path=artifact_path,
            rollback_artifact_path=rollback_path,
            metadata=metadata or {},
        )
        write_json(paths.state_path, operation.model_dump())
        write_json(
            paths.spec_path,
            {
                "workflow": workflow,
                "kind": kind,
                "spec": spec,
            },
        )
        worker_pid = self._spawn_worker(paths)
        if worker_pid is not None:
            operation = operation.model_copy(
                update={
                    "metadata": {
                        **operation.metadata,
                        _LAUNCH_MODE_METADATA_KEY: _DETACHED_WORKER_LAUNCH_MODE,
                        _WORKER_PID_METADATA_KEY: worker_pid,
                    }
                }
            )
            write_json(paths.state_path, operation.model_dump())
        return operation

    def get(self, job_id: str) -> WorkflowOperation:
        state_path = self._find_state_path(job_id)
        return self._load_operation(state_path)

    def fail(self, job_id: str, *, error: str, error_kind: str) -> WorkflowOperation:
        state_path = self._find_state_path(job_id)
        operation = self._load_operation(state_path)
        if operation.status not in {"pending", "running"}:
            raise ValueError(
                "Cannot fail workflow operation {}: status is {} "
                "(expected pending or running)".format(job_id, operation.status)
            )
        operation = operation.model_copy(
            update={
                "status": "failed",
                "finished_at": _now_iso(),
                "error": error,
                "error_kind": error_kind,
            }
        )
        write_json(state_path, operation.model_dump())
        return operation

    def delete_failed(self, job_id: str) -> WorkflowOperation:
        state_path = self._find_state_path(job_id)
        operation = self._load_operation(state_path)
        if operation.status != "failed":
            raise ValueError(
                "Cannot delete workflow operation {}: status is {} (expected failed)".format(
                    job_id, operation.status
                )
            )
        job_dir = Path(operation.job_dir)
        if job_dir != state_path.parent:
            raise ValueError(
                "Cannot delete workflow operation {}: job_dir does not match state path".format(
                    job_id
                )
            )
        shutil.rmtree(job_dir)
        return operation

    def list(self, workflow: str, kind: str | None = None) -> list[WorkflowOperation]:
        workflow_root = self.root / workflow
        if not workflow_root.is_dir():
            return []
        kinds = [kind] if kind is not None else [entry.name for entry in workflow_root.iterdir()]
        operations: list[WorkflowOperation] = []
        for current_kind in kinds:
            kind_root = workflow_root / current_kind
            if not kind_root.is_dir():
                continue
            for job_dir in sorted(kind_root.iterdir()):
                state_path = job_dir / _OPERATION_STATE
                if not state_path.is_file():
                    continue
                operations.append(self._load_operation(state_path))
        operations.sort(key=lambda record: record.started_at, reverse=True)
        return operations

    def latest(self, workflow: str, kind: str) -> WorkflowOperation | None:
        operations = self.list(workflow, kind)
        return operations[0] if operations else None

    def run_worker(self, job_dir: Path) -> None:
        state_path = job_dir / _OPERATION_STATE
        spec_path = job_dir / _OPERATION_SPEC
        operation = WorkflowOperation.model_validate(read_json(state_path))
        spec_payload = read_json(spec_path)
        if not isinstance(spec_payload, dict):
            raise ValueError(f"Invalid operation spec: {spec_path}")
        try:
            operation = operation.model_copy(update={"status": "running"})
            write_json(state_path, operation.model_dump())
            result_summary = self._dispatch(job_dir, operation, spec_payload["spec"])
            operation = operation.model_copy(
                update={
                    "status": "completed",
                    "finished_at": _now_iso(),
                    "result_summary": result_summary,
                }
            )
            write_json(state_path, operation.model_dump())
        except Exception as exc:
            operation = operation.model_copy(
                update={
                    "status": "failed",
                    "finished_at": _now_iso(),
                    "error": str(exc),
                    "error_kind": exc.__class__.__name__,
                }
            )
            write_json(state_path, operation.model_dump())

    def ensure_no_active(
        self,
        workflow: str,
        kind: str,
        *,
        metadata_key: str | None = None,
        metadata_value: str | None = None,
    ) -> None:
        for operation in self.list(workflow, kind):
            if operation.status not in {"pending", "running"}:
                continue
            if metadata_key is None:
                raise RuntimeError(f"{workflow}:{kind} is already running")
            if operation.metadata.get(metadata_key) == metadata_value:
                raise RuntimeError(f"{workflow}:{kind} is already running for {metadata_value}")

    def _dispatch(
        self,
        job_dir: Path,
        operation: WorkflowOperation,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        if operation.workflow == "standardizer":
            return self._run_standardizer(job_dir, operation, spec)
        if operation.workflow == "dataset_builder":
            return self._run_dataset_builder(job_dir, operation, spec)
        raise ValueError(f"Unsupported workflow: {operation.workflow}")

    def _run_standardizer(
        self,
        job_dir: Path,
        operation: WorkflowOperation,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        collection_root = self.environment.collection_root
        fs = self._default_fs()
        if operation.kind == "import":
            identity_map_path = Path(str(spec["identity_map_path"]))
            import_result = import_raw_batches_from_identity_map_path(
                root=collection_root,
                identity_map_path=identity_map_path,
                fs=fs,
            )
            return {
                "imported_batches": len(import_result.imported_batches),
                "imported_recordings": len(import_result.imported_recordings),
                "matched_csv_count": import_result.matched_csv_count,
                "missing_csv_count": import_result.missing_csv_count,
                "collection_paths": import_result.imported_collection_dirs,
                "batch_names": import_result.imported_batches,
                "recording_preview": [
                    recording.model_dump(exclude={"track_points"})
                    for recording in import_result.imported_recordings[:20]
                ],
            }
        if operation.kind == "scan":
            report = scan_root(collection_root)
            artifact_path = Path(operation.artifact_path or job_dir / "scan_report.json")
            write_scan_report(report, artifact_path)
            status_counts = {
                "ready": 0,
                "raw_only": 0,
                "clips_only": 0,
                "hums_only": 0,
                "empty": 0,
            }
            collections_detail = [
                {
                    "name": str(item["collection"]),
                    "clips_dir": str(item["clips_dir"]),
                    "hums_dir": str(item["hums_dir"]),
                    "raw_recordings_dir": str(item.get("raw_recordings_dir", "")),
                    "clip_count": int(item["n_clips"]),
                    "hum_count": int(item["n_hums"]),
                    "raw_recording_count": int(item.get("n_raw_recordings", 0)),
                    "has_clips": bool(item.get("has_clips", False)),
                    "has_hums": bool(item.get("has_hums", False)),
                    "has_raw_recordings": bool(item.get("has_raw_recordings", False)),
                    "status": str(item.get("status", "ready")),
                    "errors": [],
                }
                for item in report.payload["collections"]
            ]
            for item in collections_detail:
                item_status = str(item["status"])
                if item_status in status_counts:
                    status_counts[item_status] += 1
            return {
                "collections": len(collections_detail),
                "total_clips": sum(item["clip_count"] for item in collections_detail),
                "total_hums": sum(item["hum_count"] for item in collections_detail),
                "total_raw_recordings": sum(
                    item["raw_recording_count"] for item in collections_detail
                ),
                "status_counts": status_counts,
                "has_errors": False,
                "collections_detail": collections_detail,
            }
        if operation.kind == "plan":
            identity_map_path = Path(str(spec["identity_map_path"]))
            identity_map = load_identity_map(identity_map_path)
            collections: list[dict[str, Any]] = []
            total_file_renames = 0
            total_dir_renames = 0
            aggregated_ops: list[dict[str, str]] = []
            aggregated_clip_audit: list[dict[str, Any]] = []
            aggregated_hum_audit: list[dict[str, Any]] = []
            aggregated_raw_audit: list[dict[str, Any]] = []
            aggregated_recordings_updates: list[dict[str, Any]] = []
            for collection_dir in find_collection_dirs(collection_root, fs):
                (
                    collection_ops,
                    clip_audit,
                    hum_audit,
                    raw_audit,
                    recording_updates,
                ) = plan_renames_for_collection(
                    collection_dir=collection_dir,
                    identity_map=identity_map,
                    category_names=CategoryNames(),
                    fs=fs,
                )
                scan_result = scan_collection(collection_dir, CategoryNames(), fs)
                aggregated_ops.extend([{"src": op.src, "dst": op.dst} for op in collection_ops])
                aggregated_clip_audit.extend([row.__dict__ for row in clip_audit])
                aggregated_hum_audit.extend([row.__dict__ for row in hum_audit])
                aggregated_raw_audit.extend([row.__dict__ for row in raw_audit])
                aggregated_recordings_updates.extend([row.__dict__ for row in recording_updates])
                clip_renames = [
                    {
                        "src": row.old_path,
                        "dst": row.new_path,
                        "subject_token": row.subject_token_original,
                        "canonical_id": row.subject_id,
                        "date": row.date_yyyymmdd,
                        "time": row.time_hhmmss,
                        "note": row.note,
                    }
                    for row in clip_audit
                ]
                raw_recording_renames = [
                    {
                        "src": row.old_path,
                        "dst": row.new_path,
                        "recording_key": row.recording_key,
                        "kind": row.kind,
                    }
                    for row in raw_audit
                ]
                dir_renames = [
                    {"src": op.src, "dst": op.dst}
                    for op in collection_ops
                    if fs.is_dir(Path(op.src)) and Path(op.src).parent == collection_dir
                ]
                total_file_renames += sum(1 for op in collection_ops if not fs.is_dir(Path(op.src)))
                total_dir_renames += len(dir_renames)
                collections.append(
                    {
                        "name": collection_dir.name,
                        "status": scan_result.status,
                        "has_clips": scan_result.has_clips,
                        "has_hums": scan_result.has_hums,
                        "has_raw_recordings": scan_result.has_raw_recordings,
                        "clip_renames": clip_renames,
                        "raw_recording_renames": raw_recording_renames,
                        "dir_renames": dir_renames,
                    }
                )
            payload = {
                "root": str(collection_root),
                "ops": aggregated_ops,
                "audit": {
                    "clips": aggregated_clip_audit,
                    "hums": aggregated_hum_audit,
                    "raw_recordings": aggregated_raw_audit,
                },
                "recordings_updates": aggregated_recordings_updates,
            }
            artifact_path = Path(operation.artifact_path or job_dir / "plan.json")
            write_json(artifact_path, payload)
            plan_hash = _plan_hash(artifact_path)
            return {
                "plan_id": operation.job_id,
                "plan_hash": plan_hash,
                "total_file_renames": total_file_renames,
                "total_dir_renames": total_dir_renames,
                "has_collisions": False,
                "artifact_hpc_path": str(artifact_path),
                "collections": collections,
            }
        if operation.kind == "apply":
            plan_job_id = str(spec["plan_job_id"])
            plan_operation = self.get(plan_job_id)
            plan_path = Path(plan_operation.artifact_path or "")
            if not plan_path.is_file():
                raise FileNotFoundError(f"Plan artifact not found: {plan_path}")
            plan_hash = _plan_hash(plan_path)
            confirmation_phrase = str(spec["confirmation_phrase"]).strip()
            expected_phrase = f"APPLY {plan_hash}"
            if confirmation_phrase != expected_phrase:
                raise ValueError(
                    f"Confirmation phrase does not match. Expected: {expected_phrase!r}"
                )
            try:
                ops_applied = apply_rename_plan_file(plan_path)
                return {
                    "plan_id": plan_job_id,
                    "plan_hash": plan_hash,
                    "ops_applied": ops_applied,
                    "rollback_status": "not_needed",
                }
            except RollbackIncompleteError as exc:
                rollback_path = Path(operation.rollback_artifact_path or job_dir / "rollback.json")
                if exc.artifact is not None:
                    write_json(rollback_path, asdict(exc.artifact))
                raise
        if operation.kind == "index":
            identity_map_path = Path(str(spec["identity_map_path"]))
            min_quality = int(spec["min_quality"])
            artifacts_dir = job_dir / "indexes"
            index_report = build_indexes_from_identity_map_path(
                root=collection_root,
                identity_map_path=identity_map_path,
                out_dir=artifacts_dir,
                min_source_quality_to_keep=min_quality,
            )
            merged_index_path = artifacts_dir / "merged_index.json"
            shutil.copyfile(merged_index_path, self.environment.merged_index_path)
            preview_entries = index_report.merged_payload.get("entries", [])[:20]
            kept = len(index_report.merged_payload.get("entries", []))
            excluded = 0
            for payload in index_report.per_collection_payloads.values():
                for raw_entry in payload.get("entries", []):
                    entry = raw_entry if isinstance(raw_entry, dict) else {}
                    if not entry.get("keep", True):
                        excluded += 1
            return {
                "total_hums": kept + excluded,
                "kept": kept,
                "excluded": excluded,
                "min_source_quality_to_keep": min_quality,
                "generated_at": index_report.merged_payload.get("meta", {}).get(
                    "generated_at", _now_iso()
                ),
                "n_recordings": int(
                    index_report.merged_payload.get("meta", {}).get(
                        "n_recordings",
                        0,
                    )
                ),
                "n_recordings_with_sidecar": int(
                    index_report.merged_payload.get("meta", {}).get(
                        "n_recordings_with_sidecar",
                        0,
                    )
                ),
                "merged_index_hpc_path": str(merged_index_path),
                "per_collection_paths": [
                    str(artifacts_dir / collection_name / "index.json")
                    for collection_name in sorted(index_report.per_collection_payloads.keys())
                ],
                "preview_entries": preview_entries,
            }
        raise ValueError(f"Unsupported standardizer operation: {operation.kind}")

    def _run_dataset_builder(
        self,
        job_dir: Path,
        operation: WorkflowOperation,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        if operation.kind == "build":
            strategy_name = str(spec["strategy_name"])
            strategy_config = StrategyConfig.model_validate(spec["strategy_config"])
            build_result = build_dataset(
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                collection_root=self.environment.collection_root,
                merged_index_path=self.environment.merged_index_path,
                datasets_root=self.environment.datasets_root,
            )
            manifest = build_result.manifest
            summary = {
                "strategy_name": strategy_name,
                "dataset_name": strategy_name,
                "dataset_dir": str(build_result.dataset_dir),
                "n_snippets": manifest.meta.n_snippets,
                "n_target": manifest.meta.n_target,
                "n_noise": manifest.meta.n_noise,
                "splits": build_result.splits,
                "strategy_config": manifest.meta.strategy_config,
                "provenance_summary": manifest.meta.provenance_summary,
                "manual_curation_summary": manifest.meta.manual_curation_summary,
                "curated_summary": build_result.curated_summary,
            }
            _validate_dataset_build_summary(summary, manifest)
            return summary
        if operation.kind == "prepare_review":
            dataset_name = str(spec["dataset_name"])
            dataset_dir = self.environment.resolve_dataset_dir(dataset_name)
            review_result: ReviewPrepResult = prepare_review(dataset_dir)
            return {
                "dataset_name": dataset_name,
                "n_target_snippets": review_result.n_target_snippets,
                "n_noise_snippets": review_result.n_noise_snippets,
                "target_concat_wav_hpc_path": str(review_result.target_concat_wav_path),
                "noise_concat_wav_hpc_path": str(review_result.noise_concat_wav_path),
                "target_selection_table_hpc_path": str(review_result.target_selection_table_path),
                "noise_selection_table_hpc_path": str(review_result.noise_selection_table_path),
            }
        if operation.kind == "apply_review":
            dataset_name = str(spec["dataset_name"])
            target_review_table_path = Path(str(spec["target_review_table_path"]))
            noise_review_table_path = Path(str(spec["noise_review_table_path"]))
            dataset_dir = self.environment.resolve_dataset_dir(dataset_name)
            apply_result: ReviewApplyResult = apply_review(
                dataset_dir,
                target_review_table_path,
                noise_review_table_path,
            )
            return {
                "n_corrections": apply_result.n_corrections,
                "n_discarded": apply_result.n_discarded,
                "n_target": apply_result.updated_manifest.meta.n_target,
                "n_noise": apply_result.updated_manifest.meta.n_noise,
            }
        raise ValueError(f"Unsupported dataset-builder operation: {operation.kind}")

    def _spawn_worker(self, paths: OperationPaths) -> int | None:
        with (
            paths.stdout_path.open("ab") as stdout_handle,
            paths.stderr_path.open("ab") as stderr_handle,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "alpaca_pipelines.cli",
                    "_execute-operation",
                    "--job-dir",
                    str(paths.job_dir),
                ],
                start_new_session=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        return int(process.pid)

    def _find_state_path(self, job_id: str) -> Path:
        for workflow_root in self.root.iterdir() if self.root.is_dir() else []:
            for kind_root in workflow_root.iterdir():
                state_path = kind_root / job_id / _OPERATION_STATE
                if state_path.is_file():
                    return state_path
        raise FileNotFoundError(f"Workflow operation not found: {job_id}")

    def _paths(self, workflow: str, kind: str, job_id: str) -> OperationPaths:
        job_dir = self.root / workflow / kind / job_id
        return OperationPaths(
            job_dir=job_dir,
            state_path=job_dir / _OPERATION_STATE,
            spec_path=job_dir / _OPERATION_SPEC,
            stdout_path=job_dir / _RUNNER_STDOUT,
            stderr_path=job_dir / _RUNNER_STDERR,
        )

    @staticmethod
    def _default_fs() -> FileSystem:
        from alpaca_pipelines.collections.fs import _DEFAULT_FS

        return _DEFAULT_FS

    def _load_operation(self, state_path: Path) -> WorkflowOperation:
        operation = WorkflowOperation.model_validate(read_json(state_path))
        return self._reconcile_active_operation(state_path, operation)

    def _reconcile_active_operation(
        self,
        state_path: Path,
        operation: WorkflowOperation,
    ) -> WorkflowOperation:
        if operation.status not in {"pending", "running"}:
            return operation
        launch_mode = operation.metadata.get(_LAUNCH_MODE_METADATA_KEY)
        if launch_mode is None:
            return self._mark_operation_failed(
                state_path,
                operation,
                error="Workflow operation is missing detached worker launch metadata.",
                error_kind="StaleOperation",
            )
        if launch_mode != _DETACHED_WORKER_LAUNCH_MODE:
            return operation

        worker_pid = self._parse_worker_pid(operation.metadata.get(_WORKER_PID_METADATA_KEY))
        if worker_pid is None:
            return self._mark_operation_failed(
                state_path,
                operation,
                error="Detached workflow worker metadata is missing worker_pid.",
                error_kind="StaleOperation",
            )
        if self._worker_process_is_active(worker_pid, Path(operation.job_dir)):
            return operation
        # Re-read state to avoid racing with the worker's own terminal-state write.
        latest = self._read_operation_state(state_path)
        if latest is not None:
            if latest.status not in {"pending", "running"}:
                return latest
            latest_pid = self._parse_worker_pid(latest.metadata.get(_WORKER_PID_METADATA_KEY))
            if latest_pid is not None and self._worker_process_is_active(
                latest_pid, Path(latest.job_dir)
            ):
                return latest
            operation = latest
        return self._mark_operation_failed(
            state_path,
            operation,
            error="Detached workflow worker process {} is no longer running.".format(worker_pid),
            error_kind="StaleOperation",
        )

    def _read_operation_state(self, state_path: Path) -> WorkflowOperation | None:
        try:
            return WorkflowOperation.model_validate(read_json(state_path))
        except Exception:
            return None

    def _mark_operation_failed(
        self,
        state_path: Path,
        operation: WorkflowOperation,
        *,
        error: str,
        error_kind: str,
    ) -> WorkflowOperation:
        updated = operation.model_copy(
            update={
                "status": "failed",
                "finished_at": _now_iso(),
                "error": error,
                "error_kind": error_kind,
            }
        )
        write_json(state_path, updated.model_dump())
        return updated

    @staticmethod
    def _parse_worker_pid(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    @staticmethod
    def _worker_process_is_active(worker_pid: int, job_dir: Path) -> bool:
        if worker_pid <= 0:
            return False
        proc_cmdline = Path("/proc") / str(worker_pid) / "cmdline"
        if proc_cmdline.is_file():
            try:
                raw_cmdline = proc_cmdline.read_bytes().replace(b"\x00", b" ")
            except OSError:
                return False
            return str(job_dir).encode("utf-8") in raw_cmdline
        try:
            os.kill(worker_pid, 0)
        except OSError:
            return False
        return True


def _plan_hash(plan_path: Path) -> str:
    payload = json.dumps(read_json(plan_path), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]
