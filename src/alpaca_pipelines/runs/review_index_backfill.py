from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpaca_pipelines.io_utils import write_json
from alpaca_pipelines.runs.manager import RunManager
from alpaca_pipelines.runs.review_index import (
    REVIEW_INDEX_SUMMARY_FILENAME,
    build_review_index_summary_from_run_state,
)


@dataclass
class ReviewIndexBackfillSummary:
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


def _artifact_path_for_run(run_state: Any) -> Path:
    configured = getattr(run_state.outputs, "prediction_review_index_summary_path", None)
    if isinstance(configured, str) and configured:
        return Path(configured)
    predictions_dir = getattr(run_state.outputs, "predictions_dir", None)
    if not isinstance(predictions_dir, str) or not predictions_dir:
        raise ValueError("Run is missing outputs.predictions_dir")
    return Path(predictions_dir) / REVIEW_INDEX_SUMMARY_FILENAME


def _backfill_single_run(run_manager: RunManager, run_state: Any) -> bool:
    destination = _artifact_path_for_run(run_state)
    if destination.is_file():
        return False

    payload = build_review_index_summary_from_run_state(run_state)
    write_json(destination, payload)
    run_manager.update_outputs(
        run_state.run_id,
        prediction_review_index_summary_path=str(destination),
    )
    return True


def backfill_review_index_summaries(
    run_manager: RunManager,
    *,
    run_id: str | None = None,
) -> ReviewIndexBackfillSummary:
    summary = ReviewIndexBackfillSummary()
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
        candidate_runs = [
            *run_manager.list_runs(run_type="prediction", status_filter="completed"),
            *run_manager.list_runs(run_type="rf_inference", status_filter="completed"),
        ]

    summary.scanned = len(candidate_runs)
    for run_state in candidate_runs:
        current_run_id = str(run_state.run_id)
        if run_state.run_type not in ("prediction", "rf_inference"):
            summary.failed.append(
                {
                    "run_id": current_run_id,
                    "error": "Run type is {}, expected prediction|rf_inference".format(
                        run_state.run_type
                    ),
                }
            )
            continue
        if run_state.status != "completed":
            summary.skipped.append(current_run_id)
            continue
        try:
            changed = _backfill_single_run(run_manager, run_state)
        except Exception as exc:
            summary.failed.append({"run_id": current_run_id, "error": str(exc)})
            continue

        if changed:
            summary.migrated.append(current_run_id)
        else:
            summary.skipped.append(current_run_id)

    return summary
