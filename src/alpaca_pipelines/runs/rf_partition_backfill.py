from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.runs.manager import RunManager


@dataclass
class RfInferencePartitionBackfillSummary:
    scanned: int = 0
    migrated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "n_migrated": len(self.migrated),
            "n_skipped": len(self.skipped),
            "n_failed": len(self.failed),
            "migrated": self.migrated,
            "skipped": self.skipped,
            "failed": self.failed,
        }


def _partition_rf_detections(
    detections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for detection in detections:
        rf_score = detection.get("rf_score")
        if rf_score is None:
            unscored.append(detection)
        elif bool(detection.get("rf_pass")):
            accepted.append(detection)
        else:
            rejected.append(detection)
    return accepted, rejected, unscored


def _derive_partition_path(filtered_path: Path, partition: str) -> Path:
    suffix = "_rf_filtered.json"
    if filtered_path.name.endswith(suffix):
        base_stem = filtered_path.name[: -len(suffix)]
    else:
        base_stem = filtered_path.stem
    return filtered_path.with_name(f"{base_stem}_rf_{partition}.json")


def _resolve_summary_path(run_dir: Path, predictions_dir_value: str | None) -> Path:
    if isinstance(predictions_dir_value, str) and predictions_dir_value:
        return Path(predictions_dir_value) / "prediction_summary.json"
    return run_dir / "outputs" / "predictions" / "prediction_summary.json"


def _read_filtered_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"RF filtered artifact must be a JSON object: {path}")
    raw_detections = payload.get("detections")
    if not isinstance(raw_detections, list):
        raise ValueError(f"RF filtered artifact is missing detections list: {path}")
    detections: list[dict[str, Any]] = []
    for item in raw_detections:
        if not isinstance(item, dict):
            raise ValueError(f"RF filtered detection entry is malformed: {path}")
        detections.append(item)
    return payload, detections


def _backfill_single_run(run_state: Any) -> bool:
    run_dir = Path(str(run_state.run_dir))
    summary_path = _resolve_summary_path(run_dir, run_state.outputs.predictions_dir)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing prediction summary: {summary_path}")
    summary_payload = read_json(summary_path)
    if not isinstance(summary_payload, dict):
        raise ValueError(f"Prediction summary must be a JSON object: {summary_path}")

    rf_summary = summary_payload.get("rf_filter_summary")
    if not isinstance(rf_summary, dict):
        raise ValueError(f"Prediction summary missing rf_filter_summary: {summary_path}")
    raw_files = rf_summary.get("files")
    if not isinstance(raw_files, list):
        raise ValueError(f"Prediction summary missing rf_filter_summary.files list: {summary_path}")

    changed = False
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            raise ValueError(
                "rf_filter_summary.files[{}] must be an object in {}".format(index, summary_path)
            )
        filtered_raw_path = raw_file.get("rf_filtered_file")
        if not isinstance(filtered_raw_path, str) or not filtered_raw_path:
            raise ValueError(
                "rf_filter_summary.files[{}] missing rf_filtered_file in {}".format(
                    index, summary_path
                )
            )
        filtered_path = Path(filtered_raw_path)
        if not filtered_path.is_file():
            raise FileNotFoundError(f"Missing rf_filtered_file artifact: {filtered_path}")

        filtered_payload, filtered_detections = _read_filtered_artifact(filtered_path)
        accepted_detections, rejected_detections, unscored_detections = _partition_rf_detections(
            filtered_detections
        )
        for partition_name, detections_for_partition in (
            ("accepted", accepted_detections),
            ("rejected", rejected_detections),
            ("unscored", unscored_detections),
        ):
            key = f"rf_{partition_name}_file"
            artifact_path_raw = raw_file.get(key)
            if isinstance(artifact_path_raw, str) and artifact_path_raw:
                artifact_path = Path(artifact_path_raw)
            else:
                artifact_path = _derive_partition_path(filtered_path, partition_name)
                raw_file[key] = str(artifact_path)
                changed = True

            if artifact_path.is_file():
                continue
            write_json(
                artifact_path,
                {**filtered_payload, "detections": detections_for_partition},
            )
            changed = True

    if changed:
        write_json(summary_path, summary_payload)
    return changed


def backfill_rf_inference_partitions(
    run_manager: RunManager,
    *,
    run_id: str | None = None,
) -> RfInferencePartitionBackfillSummary:
    summary = RfInferencePartitionBackfillSummary()
    candidate_runs = []
    if run_id is not None:
        try:
            run_state = run_manager.find_run(run_id)
        except FileNotFoundError as exc:
            summary.failed.append({"run_id": run_id, "error": str(exc)})
            summary.scanned = 1
            return summary
        candidate_runs = [run_state]
    else:
        candidate_runs = run_manager.list_runs(run_type="rf_inference", status_filter="completed")

    summary.scanned = len(candidate_runs)
    for run_state in candidate_runs:
        current_run_id = str(run_state.run_id)
        if run_state.run_type != "rf_inference":
            summary.failed.append(
                {
                    "run_id": current_run_id,
                    "error": "Run type is {}, expected rf_inference".format(run_state.run_type),
                }
            )
            continue
        if run_state.status != "completed":
            summary.skipped.append(current_run_id)
            continue
        try:
            changed = _backfill_single_run(run_state)
        except Exception as exc:
            summary.failed.append({"run_id": current_run_id, "error": str(exc)})
            continue

        if changed:
            summary.migrated.append(current_run_id)
        else:
            summary.skipped.append(current_run_id)

    return summary
