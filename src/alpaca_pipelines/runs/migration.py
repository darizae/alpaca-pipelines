from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alpaca_pipelines.contracts import RUN_STATE_FILENAME, RunType
from alpaca_pipelines.io_utils import read_json
from alpaca_pipelines.runs.manager import RunManager

_BACKEND_META_FILENAME = "backend_meta.json"
_PIPELINE_RUN_TYPES: tuple[RunType, ...] = (
    "training",
    "prediction",
    "evaluation",
    "rf_training",
)


@dataclass
class MigrationSummary:
    migrated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    inconsistent: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "migrated": self.migrated,
            "skipped": self.skipped,
            "inconsistent": self.inconsistent,
        }


def migrate_backend_meta(
    runs_root: Path,
    run_manager: RunManager,
) -> MigrationSummary:
    summary = MigrationSummary()
    for run_type in _PIPELINE_RUN_TYPES:
        type_dir = runs_root / run_type
        if not type_dir.is_dir():
            continue
        for run_dir in sorted(type_dir.iterdir()):
            state_path = run_dir / RUN_STATE_FILENAME
            meta_path = run_dir / _BACKEND_META_FILENAME
            if meta_path.is_file() and not state_path.is_file():
                summary.inconsistent.append(str(run_dir))
                continue
            if not state_path.is_file() or not meta_path.is_file():
                summary.skipped.append(str(run_dir))
                continue
            state = run_manager.load_state(run_type, run_dir.name)
            raw_meta = read_json(meta_path)
            if not isinstance(raw_meta, dict):
                summary.inconsistent.append(str(run_dir))
                continue
            submitted_at = raw_meta.get("submitted_at")
            slurm_job_id = raw_meta.get("slurm_job_id")
            if not isinstance(submitted_at, str) or not isinstance(slurm_job_id, str):
                summary.inconsistent.append(str(run_dir))
                continue

            updated_fields: dict[str, str] = {}
            if state.submitted_at is None:
                updated_fields["submitted_at"] = submitted_at
            elif state.submitted_at != submitted_at:
                summary.inconsistent.append(str(run_dir))
                continue
            if state.slurm_job_id is None:
                updated_fields["slurm_job_id"] = slurm_job_id
            elif state.slurm_job_id != slurm_job_id:
                summary.inconsistent.append(str(run_dir))
                continue

            if not updated_fields:
                summary.skipped.append(str(run_dir))
                continue

            run_manager._persist_state(state.model_copy(update=updated_fields))
            summary.migrated.append(str(run_dir))

    if summary.inconsistent:
        raise ValueError(
            "Inconsistent backend_meta migration state for: {}".format(
                ", ".join(summary.inconsistent)
            )
        )
    return summary
