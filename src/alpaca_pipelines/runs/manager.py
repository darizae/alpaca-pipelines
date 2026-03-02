"""
Run lifecycle management.

The RunManager is responsible for creating, persisting, loading,
and transitioning pipeline runs.  It operates entirely on the
folder-based persistence under ALPACA_RUNS_ROOT.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca_pipelines.contracts import (
    EVALUATION_DIR,
    LOGS_DIR,
    MODEL_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    RUN_STATE_FILENAME,
    SLURM_DIR,
    SUMMARIES_DIR,
    RunOutputs,
    RunState,
    RunStatus,
    RunType,
)
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.paths import ensure_directory
from alpaca_pipelines.runs.state import (
    transition_to_cancelled,
    transition_to_completed,
    transition_to_failed,
    transition_to_running,
    transition_to_submitted,
)

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunManager:
    """Manages the lifecycle of pipeline runs under ALPACA_RUNS_ROOT.

    Directory structure:
        ALPACA_RUNS_ROOT/<run_type>/<run_id>/
            run_state.json
            logs/
            outputs/
                model/
                predictions/
                evaluation/
                summaries/
            slurm/
    """

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root

    def _run_dir(self, run_type: RunType, run_id: str) -> Path:
        return self.runs_root / run_type / run_id

    def _state_path(self, run_type: RunType, run_id: str) -> Path:
        return self._run_dir(run_type, run_id) / RUN_STATE_FILENAME

    def _scaffold_run_dirs(self, run_dir: Path) -> None:
        """Create the standard directory scaffold for a new run."""
        ensure_directory(run_dir / LOGS_DIR)
        ensure_directory(run_dir / OUTPUTS_DIR / MODEL_DIR)
        ensure_directory(run_dir / OUTPUTS_DIR / PREDICTIONS_DIR)
        ensure_directory(run_dir / OUTPUTS_DIR / EVALUATION_DIR)
        ensure_directory(run_dir / OUTPUTS_DIR / SUMMARIES_DIR)
        ensure_directory(run_dir / SLURM_DIR)

    def create_run(
        self,
        run_type: RunType,
        spec: dict[str, Any],
    ) -> RunState:
        """Create a new run with the given specification.

        The spec is stored as-is and is immutable after creation.
        """
        run_id = _generate_run_id()
        run_dir = self._run_dir(run_type, run_id)
        if run_dir.exists():
            raise FileExistsError("Run directory already exists: {}".format(run_dir))

        self._scaffold_run_dirs(run_dir)

        outputs = RunOutputs(
            model_dir=str(run_dir / OUTPUTS_DIR / MODEL_DIR),
            predictions_dir=str(run_dir / OUTPUTS_DIR / PREDICTIONS_DIR),
            evaluation_dir=str(run_dir / OUTPUTS_DIR / EVALUATION_DIR),
            summaries_dir=str(run_dir / OUTPUTS_DIR / SUMMARIES_DIR),
            log_dir=str(run_dir / LOGS_DIR),
        )

        state = RunState(
            run_id=run_id,
            run_type=run_type,
            status="created",
            created_at=_now_iso(),
            spec=spec,
            outputs=outputs,
            run_dir=str(run_dir),
        )

        self._persist_state(state)
        return state

    def load_state(self, run_type: RunType, run_id: str) -> RunState:
        """Load a run state from disk."""
        state_path = self._state_path(run_type, run_id)
        if not state_path.is_file():
            raise FileNotFoundError("Run state not found: {}".format(state_path))
        raw = read_json(state_path)
        if not isinstance(raw, dict):
            raise ValueError("Expected JSON object in run state: {}".format(state_path))
        return RunState.model_validate(raw)

    def find_run(self, run_id: str) -> RunState:
        """Find a run by ID, searching across all run types."""
        for run_type_candidate in ("training", "prediction", "evaluation"):
            state_path = self._state_path(run_type_candidate, run_id)  # type: ignore[arg-type]
            if state_path.is_file():
                return self.load_state(run_type_candidate, run_id)  # type: ignore[arg-type]
        raise FileNotFoundError("Run not found: {}".format(run_id))

    def mark_submitted(self, run_id: str) -> RunState:
        state = self.find_run(run_id)
        updated = transition_to_submitted(state)
        self._persist_state(updated)
        return updated

    def mark_running(self, run_id: str) -> RunState:
        state = self.find_run(run_id)
        updated = transition_to_running(state)
        self._persist_state(updated)
        return updated

    def mark_completed(self, run_id: str) -> RunState:
        state = self.find_run(run_id)
        updated = transition_to_completed(state)
        self._persist_state(updated)
        return updated

    def mark_failed(self, run_id: str, error_message: str) -> RunState:
        state = self.find_run(run_id)
        updated = transition_to_failed(state, error_message)
        self._persist_state(updated)
        return updated

    def mark_cancelled(self, run_id: str) -> RunState:
        state = self.find_run(run_id)
        updated = transition_to_cancelled(state)
        self._persist_state(updated)
        return updated

    def update_progress(
        self,
        run_id: str,
        current_epoch: int | None = None,
        total_epochs: int | None = None,
        current_phase: str | None = None,
        best_metric_value: float | None = None,
        best_metric_name: str | None = None,
    ) -> RunState:
        """Update progress tracking fields on a running job."""
        state = self.find_run(run_id)
        progress_updates: dict[str, Any] = {}
        if current_epoch is not None:
            progress_updates["current_epoch"] = current_epoch
        if total_epochs is not None:
            progress_updates["total_epochs"] = total_epochs
        if current_phase is not None:
            progress_updates["current_phase"] = current_phase
        if best_metric_value is not None:
            progress_updates["best_metric_value"] = best_metric_value
        if best_metric_name is not None:
            progress_updates["best_metric_name"] = best_metric_name

        updated_progress = state.progress.model_copy(update=progress_updates)
        updated = state.model_copy(update={"progress": updated_progress})
        self._persist_state(updated)
        return updated

    def update_outputs(self, run_id: str, **output_updates: Any) -> RunState:
        """Update output pointers on a run."""
        state = self.find_run(run_id)
        updated_outputs = state.outputs.model_copy(update=output_updates)
        updated = state.model_copy(update={"outputs": updated_outputs})
        self._persist_state(updated)
        return updated

    def list_runs(
        self,
        run_type: RunType | None = None,
        status_filter: RunStatus | None = None,
    ) -> list[RunState]:
        """List all runs, optionally filtered by type and/or status."""
        run_types: list[RunType] = (
            [run_type] if run_type is not None
            else ["training", "prediction", "evaluation"]
        )
        results: list[RunState] = []
        for rt in run_types:
            type_dir = self.runs_root / rt
            if not type_dir.is_dir():
                continue
            for run_dir in sorted(type_dir.iterdir()):
                state_path = run_dir / RUN_STATE_FILENAME
                if state_path.is_file():
                    try:
                        state = RunState.model_validate(read_json(state_path))
                        if status_filter is None or state.status == status_filter:
                            results.append(state)
                    except Exception as exc:
                        logger.warning(
                            "Corrupt run_state.json at {}: {}".format(state_path, exc)
                        )
                        continue
        return results

    def _persist_state(self, state: RunState) -> None:
        """Write run state to disk."""
        run_dir = Path(state.run_dir)
        state_path = run_dir / RUN_STATE_FILENAME
        write_json(state_path, state.model_dump())
