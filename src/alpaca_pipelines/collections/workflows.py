from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from alpaca_pipelines.collections.config import StandardizerConfig
from alpaca_pipelines.collections.contracts import IdentityMap, load_identity_map
from alpaca_pipelines.collections.fs import (
    _DEFAULT_FS,
    FileSystem,
    RollbackArtifact,
    RollbackIncompleteError,
)
from alpaca_pipelines.collections.indexing.build_index import build_collection_index, merge_indexes
from alpaca_pipelines.collections.io_utils import read_json, write_json
from alpaca_pipelines.collections.paths import CategoryNames, find_collection_dirs
from alpaca_pipelines.collections.planning.rename_plan import (
    RecordingPathUpdate,
    RenameOp,
    plan_renames_for_collection,
)
from alpaca_pipelines.collections.raw_import import RawImportResult, import_raw_batches
from alpaca_pipelines.collections.scanning import scan_collection
from alpaca_pipelines.recordings import (
    compute_recording_counts,
    load_collection_recordings,
    write_collection_recordings,
)


@dataclass(frozen=True)
class ScanReport:
    root: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class RenamePlanReport:
    root: Path
    payload: dict[str, Any]
    ops: list[RenameOp]


@dataclass(frozen=True)
class BuildIndexReport:
    root: Path
    out_dir: Path
    per_collection_payloads: dict[str, dict[str, Any]]
    merged_payload: dict[str, Any]


def scan_root(
    root: Path,
    category_names: CategoryNames | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> ScanReport:
    resolved_category_names = category_names or CategoryNames()
    collections = find_collection_dirs(root, fs)

    payload: dict[str, Any] = {"root": str(root), "collections": []}
    for collection_dir in collections:
        scan_result = scan_collection(collection_dir, resolved_category_names, fs)
        payload["collections"].append(
            {
                "collection": scan_result.collection_name,
                "clips_dir": str(
                    scan_result.clips_dir or collection_dir / resolved_category_names.clips_labelled
                ),
                "hums_dir": str(
                    scan_result.hums_dir or collection_dir / resolved_category_names.hums_segmented
                ),
                "raw_recordings_dir": str(
                    scan_result.raw_recordings_dir or collection_dir / "raw_recordings"
                ),
                "n_clips": len(scan_result.clip_files),
                "n_hums": len(scan_result.hum_files),
                "n_raw_recordings": len(scan_result.raw_recording_files),
                "has_clips": scan_result.has_clips,
                "has_hums": scan_result.has_hums,
                "has_raw_recordings": scan_result.has_raw_recordings,
                "status": scan_result.status,
            }
        )

    return ScanReport(root=root, payload=payload)


def write_scan_report(
    report: ScanReport,
    out: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> None:
    write_json(out, report.payload, fs)


def plan_rename_root(
    root: Path,
    identity_map: IdentityMap,
    category_names: CategoryNames | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> RenamePlanReport:
    resolved_category_names = category_names or CategoryNames()
    collections = find_collection_dirs(root, fs)

    ops: list[RenameOp] = []
    clip_audit_rows: list[dict[str, Any]] = []
    hum_audit_rows: list[dict[str, Any]] = []
    raw_recording_audit_rows: list[dict[str, Any]] = []
    recordings_updates: list[dict[str, Any]] = []

    for collection_dir in collections:
        (
            collection_ops,
            clip_audit,
            hum_audit,
            raw_audit,
            collection_updates,
        ) = plan_renames_for_collection(collection_dir, identity_map, resolved_category_names, fs)
        ops.extend(collection_ops)
        clip_audit_rows.extend([row.__dict__ for row in clip_audit])
        hum_audit_rows.extend([row.__dict__ for row in hum_audit])
        raw_recording_audit_rows.extend([row.__dict__ for row in raw_audit])
        recordings_updates.extend([row.__dict__ for row in collection_updates])

    payload: dict[str, Any] = {
        "root": str(root),
        "ops": [op.__dict__ for op in ops],
        "audit": {
            "clips": clip_audit_rows,
            "hums": hum_audit_rows,
            "raw_recordings": raw_recording_audit_rows,
        },
        "recordings_updates": recordings_updates,
    }

    return RenamePlanReport(root=root, payload=payload, ops=ops)


def plan_rename_root_from_identity_map_path(
    root: Path,
    identity_map_path: Path,
    category_names: CategoryNames | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> RenamePlanReport:
    identity_map = load_identity_map(identity_map_path)
    return plan_rename_root(
        root=root, identity_map=identity_map, category_names=category_names, fs=fs
    )


def write_rename_plan(
    report: RenamePlanReport,
    out: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> None:
    write_json(out, report.payload, fs)


def validate_rename_ops(ops: list[RenameOp], fs: FileSystem = _DEFAULT_FS) -> None:
    _detect_plan_collisions(ops)

    missing_sources: list[Path] = []
    for op in ops:
        src = Path(op.src)
        if not fs.exists(src):
            missing_sources.append(src)

    if missing_sources:
        formatted = "\n".join(str(p) for p in missing_sources)
        raise FileNotFoundError(f"Missing source paths:\n{formatted}")


def apply_rename_ops(ops: list[RenameOp], fs: FileSystem = _DEFAULT_FS) -> None:
    validate_rename_ops(ops, fs)

    ordered_ops = _order_ops_files_first_dirs_last(ops, fs)

    src_paths = [Path(op.src) for op in ordered_ops]
    dst_paths = [Path(op.dst) for op in ordered_ops]
    src_set = {p for p in src_paths}

    has_destination_that_is_source = any(dst in src_set for dst in dst_paths)

    if not has_destination_that_is_source:
        existing_targets = [dst for dst in dst_paths if fs.exists(dst)]

        if existing_targets:
            formatted = "\n".join(str(p) for p in existing_targets)
            raise FileExistsError(
                f"Target paths already exist (refusing to overwrite):\n{formatted}"
            )

        for op in ordered_ops:
            src = Path(op.src)
            dst = Path(op.dst)
            fs.makedirs(dst.parent)
            fs.rename(src, dst)
        return

    existing_targets_outside_plan = [
        dst for dst in dst_paths if fs.exists(dst) and dst not in src_set
    ]

    if existing_targets_outside_plan:
        formatted = "\n".join(str(p) for p in existing_targets_outside_plan)
        raise FileExistsError(f"Target paths already exist (refusing to overwrite):\n{formatted}")

    temp_by_src: dict[Path, Path] = {}
    for src in src_paths:
        temp = _make_temp_path(src)
        temp_by_src[src] = temp

    temp_paths = set(temp_by_src.values())
    for temp in temp_paths:
        if fs.exists(temp):
            raise FileExistsError(f"Temp path exists, refusing to proceed: {temp}")

    moved_to_dst: set[Path] = set()
    try:
        for op in ordered_ops:
            src = Path(op.src)
            temp = temp_by_src[src]
            fs.rename(src, temp)

        for op in ordered_ops:
            src = Path(op.src)
            temp = temp_by_src[src]
            dst = Path(op.dst)

            if fs.exists(dst):
                raise FileExistsError(f"Target exists, refusing to overwrite: {dst}")

            fs.makedirs(dst.parent)
            fs.rename(temp, dst)
            moved_to_dst.add(dst)
    except Exception as exc:
        rollback_errors: list[str] = []
        failed_rollbacks: list[tuple[str, str]] = []

        # Step 1: undo completed Phase-2 moves (dst → temp) in reverse.
        # This frees the destination paths so that Phase-1 can be undone safely.
        for op in reversed(ordered_ops):
            src = Path(op.src)
            temp = temp_by_src[src]
            dst = Path(op.dst)

            if dst not in moved_to_dst:
                continue

            if fs.exists(dst):
                try:
                    fs.rename(dst, temp)
                    moved_to_dst.discard(dst)
                except Exception as rollback_exc:
                    rollback_errors.append(f"[undo phase2] {dst} -> {temp}: {rollback_exc}")

        # Step 2: undo Phase-1 moves (temp → src) in reverse.
        for op in reversed(ordered_ops):
            src = Path(op.src)
            temp = temp_by_src[src]
            dst = Path(op.dst)

            if dst in moved_to_dst:
                # Phase-2 undo failed for this op; skip (already logged above).
                continue

            if fs.exists(temp):
                try:
                    fs.rename(temp, src)
                except Exception as rollback_exc:
                    rollback_errors.append(f"[undo phase1] {temp} -> {src}: {rollback_exc}")
                    failed_rollbacks.append((str(temp), str(src)))

        if rollback_errors:
            completed_moves = [
                (op.src, op.dst) for op in ordered_ops if Path(op.dst) in moved_to_dst
            ]
            artifact = RollbackArtifact(
                completed_moves=completed_moves,
                pending_temps=failed_rollbacks,
                rollback_errors=rollback_errors,
            )
            raise RollbackIncompleteError(
                "Rename failed and rollback was not fully successful.\n"
                "Rollback errors:\n" + "\n".join(rollback_errors),
                artifact=artifact,
            ) from exc
        raise


def apply_rename_plan_payload(
    payload: dict[str, Any],
    fs: FileSystem = _DEFAULT_FS,
) -> int:
    ops = [RenameOp(**op) for op in payload["ops"]]
    recording_updates = [
        RecordingPathUpdate(**update) for update in payload.get("recordings_updates", [])
    ]
    if not recording_updates:
        apply_rename_ops(ops, fs)
        return len(ops)

    if "root" not in payload:
        raise ValueError("Rename plan with recordings updates must include root")
    collection_root = Path(str(payload["root"]))

    snapshots = _snapshot_collection_recordings(
        collection_root=collection_root,
        recording_updates=recording_updates,
        fs=fs,
    )
    apply_rename_ops(ops, fs)
    try:
        _apply_recordings_updates(
            collection_root=collection_root,
            recording_updates=recording_updates,
            fs=fs,
        )
    except Exception as exc:
        rollback_errors = _restore_collection_recordings(snapshots=snapshots, fs=fs)
        try:
            reverse_ops = [RenameOp(src=op.dst, dst=op.src) for op in reversed(ops)]
            if reverse_ops:
                apply_rename_ops(reverse_ops, fs)
        except RollbackIncompleteError as rollback_exc:
            artifact = rollback_exc.artifact
            artifact.rollback_errors.extend(rollback_errors)
            raise RollbackIncompleteError(
                "Rename/apply failed and rollback was not fully successful.",
                artifact=artifact,
            ) from exc
        except Exception as rollback_exc:
            rollback_errors.append(f"[rename rollback] {rollback_exc}")

        if rollback_errors:
            raise RollbackIncompleteError(
                "Rename/apply failed and rollback was not fully successful.\n"
                "Rollback errors:\n" + "\n".join(rollback_errors),
                artifact=RollbackArtifact(rollback_errors=rollback_errors),
            ) from exc
        raise RuntimeError(
            "Failed to update recordings.json after applying rename plan. "
            "Filesystem changes were rolled back."
        ) from exc
    return len(ops)


def apply_rename_plan_file(
    plan_path: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> int:
    payload = read_json(plan_path, fs)
    if not isinstance(payload, dict):
        raise ValueError(f"Rename plan must be a JSON object: {plan_path}")
    return apply_rename_plan_payload(payload, fs)


def _snapshot_collection_recordings(
    *,
    collection_root: Path,
    recording_updates: list[RecordingPathUpdate],
    fs: FileSystem,
) -> dict[Path, str | None]:
    snapshots: dict[Path, str | None] = {}
    collections = {update.collection for update in recording_updates}
    for collection_name in collections:
        recordings_path = collection_root / collection_name / "recordings.json"
        snapshots[recordings_path] = (
            fs.read_text(recordings_path) if fs.exists(recordings_path) else None
        )
    return snapshots


def _apply_recordings_updates(
    *,
    collection_root: Path,
    recording_updates: list[RecordingPathUpdate],
    fs: FileSystem,
) -> None:
    updates_by_collection: dict[str, dict[str, RecordingPathUpdate]] = {}
    for update in recording_updates:
        updates_by_collection.setdefault(update.collection, {})[update.recording_key] = update

    for collection_name, updates_by_key in updates_by_collection.items():
        collection_dir = collection_root / collection_name
        recordings = load_collection_recordings(collection_dir, fs)
        if not recordings:
            raise ValueError(
                f"{collection_name}: recordings.json missing or empty while applying updates"
            )

        updated_recordings = []
        for recording in recordings:
            planned_update = updates_by_key.get(recording.key)
            if planned_update is None:
                updated_recordings.append(recording)
                continue
            updated_recordings.append(
                recording.model_copy(
                    update={
                        "wav_path": planned_update.wav_path,
                        "csv_path": planned_update.csv_path,
                    }
                )
            )

        missing_keys = set(updates_by_key) - {recording.key for recording in recordings}
        if missing_keys:
            raise ValueError(
                f"{collection_name}: recordings.json missing update keys: {sorted(missing_keys)}"
            )
        write_collection_recordings(collection_dir, updated_recordings, fs)


def _restore_collection_recordings(
    *,
    snapshots: dict[Path, str | None],
    fs: FileSystem,
) -> list[str]:
    rollback_errors: list[str] = []
    for recordings_path, original_content in snapshots.items():
        try:
            if original_content is None:
                if fs.exists(recordings_path):
                    fs.unlink(recordings_path)
            else:
                fs.write_text(recordings_path, original_content)
        except Exception as exc:
            rollback_errors.append(f"[recordings rollback] {recordings_path}: {exc}")
    return rollback_errors


def build_indexes(
    root: Path,
    identity_map: IdentityMap,
    out_dir: Path,
    min_source_quality_to_keep: int | None = None,
    category_names: CategoryNames | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> BuildIndexReport:
    resolved_category_names = category_names or CategoryNames()
    config = StandardizerConfig(min_source_quality_to_keep=min_source_quality_to_keep)

    collections = find_collection_dirs(root, fs)

    fs.makedirs(out_dir)

    per_collection_payloads: dict[str, dict[str, Any]] = {}
    indexes: list[dict[str, Any]] = []

    for collection_dir in collections:
        scan_result = scan_collection(collection_dir, resolved_category_names, fs)
        if scan_result.hums_dir is not None:
            if scan_result.hums_dir.name != resolved_category_names.hums_segmented:
                raise ValueError(
                    f"{collection_dir.name}: hums dir not standardized "
                    f"(expected {resolved_category_names.hums_segmented})"
                )
            index_payload = build_collection_index(
                persistence_root=root,
                collection_dir=collection_dir,
                hums_dir=scan_result.hums_dir,
                identity_map=identity_map,
                config=config,
                fs=fs,
            )
        else:
            collection_recordings = load_collection_recordings(collection_dir, fs)
            n_recordings, n_recordings_with_sidecar = compute_recording_counts(
                collection_recordings
            )
            index_payload = {
                "meta": {
                    "collection": collection_dir.name,
                    "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "n_hums": 0,
                    "n_recordings": n_recordings,
                    "n_recordings_with_sidecar": n_recordings_with_sidecar,
                    "min_source_quality_to_keep": config.min_source_quality_to_keep,
                    "persistence_root_contract": "hum_path is relative to ALPACA_COLLECTION_ROOT",
                },
                "entries": [],
                "recordings": [recording.model_dump() for recording in collection_recordings],
            }
        per_collection_payloads[collection_dir.name] = index_payload
        indexes.append(index_payload)

        out_path = out_dir / collection_dir.name / "index.json"
        write_json(out_path, index_payload, fs)

    merged_payload = merge_indexes(indexes)
    merged_path = out_dir / "merged_index.json"
    write_json(merged_path, merged_payload, fs)

    return BuildIndexReport(
        root=root,
        out_dir=out_dir,
        per_collection_payloads=per_collection_payloads,
        merged_payload=merged_payload,
    )


def build_indexes_from_identity_map_path(
    root: Path,
    identity_map_path: Path,
    out_dir: Path,
    min_source_quality_to_keep: int | None = None,
    category_names: CategoryNames | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> BuildIndexReport:
    identity_map = load_identity_map(identity_map_path)
    return build_indexes(
        root=root,
        identity_map=identity_map,
        out_dir=out_dir,
        min_source_quality_to_keep=min_source_quality_to_keep,
        category_names=category_names,
        fs=fs,
    )


def import_raw_batches_from_identity_map_path(
    root: Path,
    identity_map_path: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> RawImportResult:
    identity_map = load_identity_map(identity_map_path)
    return import_raw_batches(root, identity_map, fs)


def _detect_plan_collisions(ops: list[RenameOp]) -> None:
    dst_to_src: dict[str, str] = {}
    seen_src: set[str] = set()

    for op in ops:
        if op.src in seen_src:
            raise ValueError(f"Duplicate src in plan: {op.src}")
        seen_src.add(op.src)

        existing = dst_to_src.get(op.dst)
        if existing is not None and existing != op.src:
            raise ValueError(
                f"Destination collision in plan: {op.dst} from {existing} and {op.src}"
            )
        dst_to_src[op.dst] = op.src


def _order_ops_files_first_dirs_last(ops: list[RenameOp], fs: FileSystem) -> list[RenameOp]:
    file_ops = [op for op in ops if not fs.is_dir(Path(op.src))]
    dir_ops = [op for op in ops if fs.is_dir(Path(op.src))]
    return [*file_ops, *dir_ops]


def _make_temp_path(src: Path) -> Path:
    digest = sha256(str(src).encode("utf-8")).hexdigest()[:16]
    suffix = f".tmp_alpaca_pipelines.collections_{digest}"
    return src.with_name(f"{src.name}{suffix}")
