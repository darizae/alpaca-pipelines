"""
Run state machine and transition logic.

State transitions:
    created → submitted → running → completed
                                  → failed
    created → cancelled
    submitted → cancelled
"""

from __future__ import annotations

from datetime import datetime, timezone

from alpaca_pipelines.contracts import RunState, RunStatus

_VALID_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    "created": frozenset({"submitted", "cancelled", "running"}),
    "submitted": frozenset({"running", "cancelled"}),
    "running": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def _validate_transition(current: RunStatus, target: RunStatus) -> None:
    """Validate that a state transition is allowed."""
    allowed = _VALID_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ValueError(
            "Invalid state transition: {} → {} (allowed: {})".format(
                current, target, sorted(allowed)
            )
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def transition_to_submitted(state: RunState) -> RunState:
    """Transition a run from created to submitted."""
    _validate_transition(state.status, "submitted")
    return state.model_copy(update={"status": "submitted", "submitted_at": _now_iso()})


def transition_to_running(state: RunState) -> RunState:
    """Transition a run to running."""
    _validate_transition(state.status, "running")
    return state.model_copy(update={"status": "running", "started_at": _now_iso()})


def transition_to_completed(state: RunState) -> RunState:
    """Transition a run to completed."""
    _validate_transition(state.status, "completed")
    return state.model_copy(update={"status": "completed", "completed_at": _now_iso()})


def transition_to_failed(state: RunState, error_message: str) -> RunState:
    """Transition a run to failed with an error message."""
    _validate_transition(state.status, "failed")
    return state.model_copy(
        update={
            "status": "failed",
            "completed_at": _now_iso(),
            "error_message": error_message,
        }
    )


def transition_to_cancelled(state: RunState) -> RunState:
    """Transition a run to cancelled."""
    _validate_transition(state.status, "cancelled")
    return state.model_copy(update={"status": "cancelled", "completed_at": _now_iso()})
