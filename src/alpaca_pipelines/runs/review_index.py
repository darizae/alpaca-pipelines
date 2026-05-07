from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca_pipelines.io_utils import read_json, write_json

REVIEW_INDEX_SUMMARY_FILENAME = "review_index_summary.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_summary_file_entries(raw_files: Any, summary_path: Path) -> list[dict[str, Any]]:
    if not isinstance(raw_files, list):
        raise ValueError("prediction_summary.json missing 'files' list: {}".format(summary_path))

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, dict):
            raise ValueError(
                "prediction_summary.json files[{}] must be an object: {}".format(
                    index, summary_path
                )
            )
        audio_file = entry.get("audio_file")
        if not isinstance(audio_file, str) or not audio_file:
            raise ValueError(
                "prediction_summary.json files[{}] has invalid audio_file: {}".format(
                    index, summary_path
                )
            )
        n_windows = entry.get("n_windows", 0)
        n_detections = entry.get("n_detections", 0)
        if not isinstance(n_windows, int):
            raise ValueError(
                "prediction_summary.json files[{}] has invalid n_windows: {}".format(
                    index, summary_path
                )
            )
        if not isinstance(n_detections, int):
            raise ValueError(
                "prediction_summary.json files[{}] has invalid n_detections: {}".format(
                    index, summary_path
                )
            )
        normalized.append(
            {
                "audio_file": audio_file,
                "n_windows": n_windows,
                "n_detections": n_detections,
            }
        )
    return normalized


def _rf_partition_totals(rf_filter_summary: dict[str, Any] | None) -> dict[str, int] | None:
    if rf_filter_summary is None:
        return None
    accepted = rf_filter_summary.get("rf_passed")
    rejected = rf_filter_summary.get("rf_rejected")
    unscored = rf_filter_summary.get("rf_unscored")
    if (
        not isinstance(accepted, int)
        or not isinstance(rejected, int)
        or not isinstance(unscored, int)
    ):
        return None
    return {
        "accepted": accepted,
        "rejected": rejected,
        "unscored": unscored,
    }


def build_review_index_summary(
    *,
    run_id: str,
    run_type: str,
    predictions_dir: Path,
    prediction_summary: dict[str, Any],
) -> dict[str, Any]:
    files = _normalize_summary_file_entries(
        prediction_summary.get("files"),
        predictions_dir / "prediction_summary.json",
    )
    rf_filter_summary = prediction_summary.get("rf_filter_summary")
    if rf_filter_summary is not None and not isinstance(rf_filter_summary, dict):
        raise ValueError("prediction_summary.json has invalid rf_filter_summary object")

    per_tape: list[dict[str, Any]] = []
    rf_file_partitions: dict[str, dict[str, int]] = {}
    if isinstance(rf_filter_summary, dict):
        raw_rf_files = rf_filter_summary.get("files")
        if isinstance(raw_rf_files, list):
            for rf_entry in raw_rf_files:
                if not isinstance(rf_entry, dict):
                    continue
                audio_file = rf_entry.get("audio_file")
                if not isinstance(audio_file, str) or not audio_file:
                    continue
                passed = rf_entry.get("rf_passed")
                rejected = rf_entry.get("rf_rejected")
                unscored = rf_entry.get("rf_unscored")
                if (
                    isinstance(passed, int)
                    and isinstance(rejected, int)
                    and isinstance(unscored, int)
                ):
                    rf_file_partitions[audio_file] = {
                        "accepted": passed,
                        "rejected": rejected,
                        "unscored": unscored,
                    }

    for file_entry in files:
        row = dict(file_entry)
        partitions = rf_file_partitions.get(file_entry["audio_file"])
        if partitions is not None:
            row["rf_partitions"] = partitions
        per_tape.append(row)

    summary: dict[str, Any] = {
        "generated_at": _now_iso(),
        "run_id": run_id,
        "run_type": run_type,
        "predictions_dir": str(predictions_dir),
        "prediction_summary_path": str(predictions_dir / "prediction_summary.json"),
        "rf_filtered": bool(prediction_summary.get("rf_filtered", False)),
        "n_files": len(files),
        "total_detections": int(prediction_summary.get("total_detections", 0)),
        "files": per_tape,
    }
    partitions = _rf_partition_totals(
        rf_filter_summary if isinstance(rf_filter_summary, dict) else None
    )
    if partitions is not None:
        summary["rf_partition_totals"] = partitions
    return summary


def build_review_index_summary_from_run_state(run_state: Any) -> dict[str, Any]:
    predictions_dir_raw = getattr(run_state.outputs, "predictions_dir", None)
    if not isinstance(predictions_dir_raw, str) or not predictions_dir_raw:
        raise ValueError("Run is missing outputs.predictions_dir")
    predictions_dir = Path(predictions_dir_raw)
    summary_path = predictions_dir / "prediction_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Missing prediction summary: {}".format(summary_path))
    payload = read_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError("Prediction summary must be a JSON object: {}".format(summary_path))
    return build_review_index_summary(
        run_id=str(run_state.run_id),
        run_type=str(run_state.run_type),
        predictions_dir=predictions_dir,
        prediction_summary=payload,
    )


def write_review_index_summary(
    *,
    run_id: str,
    run_type: str,
    predictions_dir: Path,
    prediction_summary: dict[str, Any],
) -> Path:
    payload = build_review_index_summary(
        run_id=run_id,
        run_type=run_type,
        predictions_dir=predictions_dir,
        prediction_summary=prediction_summary,
    )
    destination = predictions_dir / REVIEW_INDEX_SUMMARY_FILENAME
    write_json(destination, payload)
    return destination
