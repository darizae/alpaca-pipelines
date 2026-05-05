"""Executor for standalone RF inference runs."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.rf.executor import apply_rf_filter
from alpaca_pipelines.rf_inference.config import RfInferenceRunSpec
from alpaca_pipelines.runs.manager import RunManager

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_prediction_output_path(predictions_dir: Path, audio_file: str) -> Path:
    stem = Path(audio_file).stem
    canonical = predictions_dir / f"{stem}.json"
    if canonical.is_file():
        return canonical
    hashed_candidates = sorted(
        path
        for path in predictions_dir.glob(f"{stem}_*.json")
        if path.name != "prediction_summary.json"
    )
    if len(hashed_candidates) == 1:
        return hashed_candidates[0]
    raise FileNotFoundError(
        "Prediction output JSON is missing or ambiguous for audio_file={!r} in {}".format(
            audio_file, predictions_dir
        )
    )


def execute_rf_inference(
    run_state: RunState,
    environment: PipelineEnvironment,
    run_manager: RunManager,
) -> RunState:
    del environment
    run_state = run_manager.mark_running(run_state.run_id)
    spec = RfInferenceRunSpec.model_validate(run_state.spec)
    predictions_dir = Path(run_state.outputs.predictions_dir or "")
    if not predictions_dir.is_dir():
        raise RuntimeError("rf_inference run is missing outputs.predictions_dir")

    source_predictions_dir = Path(spec.source_predictions_dir)
    source_summary_path = source_predictions_dir / "prediction_summary.json"
    if not source_summary_path.is_file():
        raise FileNotFoundError(
            "Source prediction summary not found: {}".format(source_summary_path)
        )

    source_summary = read_json(source_summary_path)
    if not isinstance(source_summary, dict):
        raise ValueError("Source prediction summary must be a JSON object")
    source_files = source_summary.get("files")
    if not isinstance(source_files, list):
        raise ValueError("Source prediction summary is missing a files list")
    if source_summary.get("rf_filtered"):
        raise ValueError("Source prediction run is already RF filtered")

    prediction_progress: dict[str, Any] = {
        "stage": "initializing",
        "files_total": None,
        "files_completed": 0,
        "current_file": None,
        "current_file_windows_total": None,
        "current_file_windows_completed": None,
        "detections_so_far": 0,
        "updated_at": _now_iso(),
    }

    def _emit_progress(*, force: bool = False, **updates: Any) -> None:
        del force
        nonlocal prediction_progress
        prediction_progress = {
            **prediction_progress,
            **updates,
            "updated_at": _now_iso(),
        }
        run_manager.update_progress(
            run_state.run_id,
            current_phase=str(prediction_progress["stage"]),
            prediction=prediction_progress,
        )

    _emit_progress(force=True, stage="initializing")

    prediction_inputs: list[dict[str, str]] = []
    normalized_files: list[dict[str, Any]] = []
    try:
        _emit_progress(force=True, stage="resolving_inputs")
        files_total = len(source_files)
        _emit_progress(force=True, files_total=files_total, files_completed=0, detections_so_far=0)
        for file_entry in source_files:
            if not isinstance(file_entry, dict):
                raise ValueError("Invalid file entry in source prediction summary")
            audio_file = file_entry.get("audio_file")
            if not isinstance(audio_file, str) or not audio_file:
                raise ValueError("Source prediction summary contains an invalid audio_file value")
            _emit_progress(stage="resolving_inputs", current_file=audio_file)
            source_output_path = _resolve_prediction_output_path(source_predictions_dir, audio_file)
            destination_output_path = predictions_dir / source_output_path.name
            shutil.copyfile(source_output_path, destination_output_path)
            prediction_inputs.append(
                {"audio_file": audio_file, "prediction_file": str(destination_output_path)}
            )
            normalized_files.append(
                {
                    "audio_file": audio_file,
                    "n_windows": file_entry.get("n_windows", 0),
                    "n_detections": file_entry.get("n_detections", 0),
                }
            )

        _emit_progress(force=True, stage="rf_filtering")
        rf_filter_summary = apply_rf_filter(
            prediction_inputs=prediction_inputs,
            rf_model_path=spec.rf_model_path,
            rf_threshold=spec.rf_threshold,
            rf_feature_config=spec.rf_feature_config.model_dump(),
            prediction_logger=logger,
        )
        files = rf_filter_summary.get("files")
        if isinstance(files, list):
            cumulative_kept = 0
            for index, file_summary in enumerate(files):
                if not isinstance(file_summary, dict):
                    continue
                audio_file = file_summary.get("audio_file")
                rf_passed = file_summary.get("rf_passed")
                if isinstance(rf_passed, int):
                    cumulative_kept += rf_passed
                _emit_progress(
                    stage="rf_filtering",
                    files_completed=index + 1,
                    current_file=audio_file if isinstance(audio_file, str) else None,
                    detections_so_far=cumulative_kept,
                )

        _emit_progress(force=True, stage="writing_summary", current_file=None)

        summary = {
            "run_id": run_state.run_id,
            "source_prediction_run_id": spec.source_prediction_run_id,
            "source_predictions_dir": spec.source_predictions_dir,
            "rf_training_run_id": spec.rf_training_run_id,
            "rf_model_path": spec.rf_model_path,
            "rf_threshold": spec.rf_threshold,
            "n_files": len(normalized_files),
            "total_detections": source_summary.get("total_detections", 0),
            "detection_threshold": source_summary.get("detection_threshold"),
            "rf_filtered": True,
            "rf_filter_summary": rf_filter_summary,
            "files": normalized_files,
        }
        write_json(predictions_dir / "prediction_summary.json", summary)
        _emit_progress(
            force=True,
            stage="completed",
            files_completed=len(normalized_files),
            current_file=None,
            current_file_windows_total=None,
            current_file_windows_completed=None,
            detections_so_far=summary["total_detections"],
        )
        run_manager.update_outputs(run_state.run_id, rf_filtered=True)
        return run_manager.mark_completed(run_state.run_id)
    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        logger.error("RF inference run failed: %s", error_message)
        run_manager.mark_failed(run_state.run_id, error_message)
        raise
