"""
Public API surface for alpaca-pipelines.

This is the primary interface that the future backend app will drive.
All operations are available as methods on ``PipelineAPI``.
The CLI is a thin wrapper around this class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState, RunStatus, RunType
from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.runs.manager import RunManager
from alpaca_pipelines.slurm.config import SlurmConfig
from alpaca_pipelines.slurm.generator import generate_slurm_script
from alpaca_pipelines.training.config import TrainingRunSpec


class PipelineAPI:
    """Programmatic API for pipeline orchestration.

    The backend creates one instance per request cycle (or as a singleton)
    and uses it to create, execute, poll, and inspect runs.

    Parameters
    ----------
    environment:
        Validated pipeline environment. Use ``PipelineEnvironment.from_env()``
        for CLI/Makefile usage, or ``PipelineEnvironment.from_explicit(...)``
        for backend/API usage.
    """

    def __init__(self, environment: PipelineEnvironment) -> None:
        environment.validate()
        self.environment = environment
        self.run_manager = RunManager(environment.runs_root)

    # ------------------------------------------------------------------
    # Run creation
    # ------------------------------------------------------------------

    def create_training_run(self, spec: TrainingRunSpec) -> RunState:
        """Create a new training run from a specification."""
        self.environment.resolve_dataset_dir(spec.dataset_name)
        return self.run_manager.create_run(
            run_type="training",
            spec=spec.to_spec_dict(),
        )

    def create_prediction_run(self, spec: PredictionRunSpec) -> RunState:
        """Create a new prediction run from a specification."""
        if spec.mode == "dataset" and spec.dataset_name is not None:
            self.environment.resolve_dataset_dir(spec.dataset_name)
        if not Path(spec.model_path).is_file():
            raise FileNotFoundError("Model file not found: {}".format(spec.model_path))
        return self.run_manager.create_run(
            run_type="prediction",
            spec=spec.to_spec_dict(),
        )

    def create_evaluation_run(self, spec: EvaluationRunSpec) -> RunState:
        """Create a new evaluation run from a specification."""
        self.environment.resolve_dataset_dir(spec.dataset_name)
        return self.run_manager.create_run(
            run_type="evaluation",
            spec=spec.to_spec_dict(),
        )

    # ------------------------------------------------------------------
    # Run execution
    # ------------------------------------------------------------------

    def execute_run(self, run_id: str) -> RunState:
        """Execute a run by its ID.

        This is the main entry point for SLURM jobs and the CLI.
        It dispatches to the appropriate executor based on run type.
        """
        run_state = self.run_manager.find_run(run_id)

        if run_state.status not in ("created", "submitted"):
            raise ValueError(
                "Cannot execute run {}: status is {} (expected created or submitted)".format(
                    run_id, run_state.status
                )
            )

        if run_state.run_type == "training":
            from alpaca_pipelines.training.executor import execute_training

            return execute_training(run_state, self.environment, self.run_manager)

        elif run_state.run_type == "prediction":
            from alpaca_pipelines.prediction.executor import execute_prediction

            return execute_prediction(run_state, self.environment, self.run_manager)

        elif run_state.run_type == "evaluation":
            from alpaca_pipelines.evaluation.executor import execute_evaluation

            return execute_evaluation(run_state, self.environment, self.run_manager)

        raise ValueError("Unknown run type: {}".format(run_state.run_type))

    # ------------------------------------------------------------------
    # Run status and inspection
    # ------------------------------------------------------------------

    def get_run_status(self, run_id: str) -> RunState:
        """Poll the current status of a run."""
        return self.run_manager.find_run(run_id)

    def list_runs(
        self,
        run_type: RunType | None = None,
        status_filter: RunStatus | None = None,
    ) -> list[RunState]:
        """List all runs, optionally filtered."""
        return self.run_manager.list_runs(run_type=run_type, status_filter=status_filter)

    def cancel_run(self, run_id: str) -> RunState:
        """Cancel a run (only from created or submitted state)."""
        return self.run_manager.mark_cancelled(run_id)

    def get_run_outputs(self, run_id: str) -> dict[str, Any]:
        """Get the output pointers for a completed run."""
        run_state = self.run_manager.find_run(run_id)
        return run_state.outputs.model_dump()

    def get_run_progress(self, run_id: str) -> dict[str, Any]:
        """Get the progress tracking for a running job."""
        run_state = self.run_manager.find_run(run_id)
        return run_state.progress.model_dump()

    # ------------------------------------------------------------------
    # SLURM integration
    # ------------------------------------------------------------------

    def generate_slurm_script(
        self,
        run_id: str,
        slurm_config: SlurmConfig | None = None,
    ) -> Path:
        """Generate a SLURM batch script for a run."""
        run_state = self.run_manager.find_run(run_id)
        if slurm_config is None:
            slurm_config = SlurmConfig()

        environment_vars = {
            "ALPACA_COLLECTION_ROOT": str(self.environment.collection_root),
            "ALPACA_MERGED_INDEX": str(self.environment.merged_index_path),
            "ALPACA_DATASETS_ROOT": str(self.environment.datasets_root),
            "ALPACA_RUNS_ROOT": str(self.environment.runs_root),
        }

        return generate_slurm_script(
            run_state=run_state,
            slurm_config=slurm_config,
            environment_vars=environment_vars,
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def export_detections_to_selection_table(
        self,
        prediction_run_id: str,
        audio_file_stem: str,
        output_path: Path,
        freq_low_hz: int = 0,
        freq_high_hz: int = 4000,
        use_rf_filtered: bool = False,
    ) -> Path:
        """Export detections from a prediction run to Raven selection table format."""
        from alpaca_pipelines.postprocessing.executor import (
            export_detections_to_selection_table,
        )

        run_state = self.run_manager.find_run(prediction_run_id)
        predictions_dir = Path(run_state.run_dir) / "outputs" / "predictions"

        suffix = "_rf_filtered" if use_rf_filtered else ""
        predictions_path = predictions_dir / "{}{}.json".format(audio_file_stem, suffix)

        return export_detections_to_selection_table(
            predictions_path=predictions_path,
            output_path=output_path,
            freq_low_hz=freq_low_hz,
            freq_high_hz=freq_high_hz,
            use_rf_filtered=use_rf_filtered,
        )

    def aggregate_evaluations(
        self,
        evaluation_run_ids: list[str],
        output_path: Path,
    ) -> dict[str, Any]:
        """Aggregate evaluation results from multiple runs."""
        from alpaca_pipelines.postprocessing.executor import aggregate_evaluation_results

        evaluation_dirs = []
        for run_id in evaluation_run_ids:
            run_state = self.run_manager.find_run(run_id)
            evaluation_dirs.append(Path(run_state.run_dir) / "outputs" / "evaluation")

        return aggregate_evaluation_results(evaluation_dirs, output_path)
