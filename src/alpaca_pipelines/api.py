"""
Public API surface for alpaca-pipelines.

This is the primary interface that the backend app drives.
All operations are available as methods on ``PipelineAPI``.
The CLI is a thin wrapper around this class.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState, RunStatus, RunType
from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.prediction.review import PredictionReviewSpectrogramConfig
from alpaca_pipelines.prediction.review.curated import (
    list_curated_prediction_categories,
    materialize_curated_prediction_examples,
    migrate_legacy_curated_prediction_sources,
)
from alpaca_pipelines.rf.config import RfFeatureConfig
from alpaca_pipelines.rf_inference.config import RfInferenceRunSpec
from alpaca_pipelines.rf_training.config import RfTrainingRunSpec
from alpaca_pipelines.runs.manager import RunManager
from alpaca_pipelines.runs.migration import MigrationSummary, migrate_backend_meta
from alpaca_pipelines.slurm.config import SlurmConfig
from alpaca_pipelines.slurm.generator import generate_slurm_script
from alpaca_pipelines.training.config import TrainingRunSpec
from alpaca_pipelines.workflow_ops import WorkflowOperationManager


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
        self.workflow_ops = WorkflowOperationManager(environment)

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
        if spec.mode == "collection":
            for collection_name in spec.collection_names:
                collection_dir = self.environment.collection_root / collection_name
                if not collection_dir.is_dir():
                    raise FileNotFoundError(
                        "Collection directory not found: {}".format(collection_dir)
                    )
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

    def create_rf_training_run(self, spec: RfTrainingRunSpec) -> RunState:
        """Create a new RF training run from a specification."""
        self.environment.resolve_dataset_dir(spec.dataset_name)
        return self.run_manager.create_run(
            run_type="rf_training",
            spec=spec.to_spec_dict(),
        )

    def create_rf_inference_run(self, spec: RfInferenceRunSpec) -> RunState:
        """Create a new standalone RF inference run from a specification."""
        if not Path(spec.source_predictions_dir).is_dir():
            raise FileNotFoundError(
                "Source predictions directory not found: {}".format(spec.source_predictions_dir)
            )
        if not Path(spec.rf_model_path).is_file():
            raise FileNotFoundError("RF model file not found: {}".format(spec.rf_model_path))
        return self.run_manager.create_run(
            run_type="rf_inference",
            spec=spec.model_dump(),
        )

    def import_rf_training_run(
        self,
        *,
        bundle_dir: Path,
        run_name: str = "",
        source_label: str | None = None,
    ) -> RunState:
        """Import an externally trained RF bundle as a completed RF training run."""
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"RF import bundle directory not found: {bundle_dir}")

        required_files = {
            "model": bundle_dir / "model.joblib",
            "feature_params": bundle_dir / "feature_params.json",
            "feature_columns": bundle_dir / "feature_columns.txt",
            "metrics": bundle_dir / "metrics.json",
        }
        for label, path in required_files.items():
            if not path.is_file():
                raise FileNotFoundError(f"RF import bundle missing required {label} file: {path}")

        feature_params = read_json(required_files["feature_params"])
        if not isinstance(feature_params, dict):
            raise ValueError("feature_params.json must be a JSON object")
        feature_payload = dict(feature_params)
        feature_payload.setdefault("pad_mode", "constant")
        feature_config = RfFeatureConfig.model_validate(feature_payload).model_dump()

        metrics = read_json(required_files["metrics"])
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json must be a JSON object")

        params_path = bundle_dir / "params.json"
        params: dict[str, Any] = {}
        if params_path.is_file():
            params_payload = read_json(params_path)
            if not isinstance(params_payload, dict):
                raise ValueError("params.json must be a JSON object")
            params = params_payload

        feature_names = [
            line.strip()
            for line in required_files["feature_columns"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not feature_names:
            raise ValueError("feature_columns.txt must contain at least one feature")

        def _as_float(payload: dict[str, Any], key: str) -> float | None:
            if key not in payload or payload[key] is None:
                return None
            try:
                return float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metrics.json field {key!r} must be numeric") from exc

        def _as_int(payload: dict[str, Any], key: str) -> int:
            value = payload.get(key, 0)
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metrics.json field {key!r} must be an integer") from exc

        rf_threshold = _as_float(metrics, "decision_threshold")
        if rf_threshold is None:
            rf_threshold = 0.4
        if rf_threshold < 0.0 or rf_threshold > 1.0:
            raise ValueError("metrics.json decision_threshold must be between 0 and 1")

        tp = _as_int(metrics, "tp")
        tn = _as_int(metrics, "tn")
        fp = _as_int(metrics, "fp")
        fn = _as_int(metrics, "fn")
        n_val_samples = tp + tn + fp + fn

        run_spec: dict[str, Any] = {
            "dataset_name": "imported_external",
            "positive_class": "target",
            "run_name": run_name,
            "rf_threshold": rf_threshold,
            "feature_config": feature_config,
            "import_source": "rf_sandbox_bundle",
            "source_bundle_dir": str(bundle_dir),
        }
        if source_label:
            run_spec["source_label"] = source_label

        run_state = self.run_manager.create_run(run_type="rf_training", spec=run_spec)
        run_state = self.run_manager.mark_running(run_state.run_id)
        run_dir = Path(run_state.run_dir)
        model_dir = run_dir / "outputs" / "model"
        report_path = run_dir / "outputs" / "summaries" / "rf_training_report.json"
        rf_model_path = model_dir / "rf_model.joblib"

        try:
            rf_model_path.write_bytes(required_files["model"].read_bytes())

            write_json(
                model_dir / "rf_model_metadata.json",
                {
                    "feature_family": "rf_v1",
                    "feature_names": feature_names,
                    "rf_threshold": rf_threshold,
                    "feature_config": feature_config,
                },
            )

            report_metrics: dict[str, Any] = {"classification_report": {}}
            for metric_name in ("accuracy", "f1", "precision", "recall", "roc_auc"):
                metric_value = _as_float(metrics, metric_name)
                if metric_value is not None:
                    report_metrics[metric_name] = round(metric_value, 6)

            report = {
                "run_id": run_state.run_id,
                "dataset_name": "imported_external",
                "positive_class": "target",
                "class_to_index": {"noise": 0, "target": 1},
                "train": {
                    "n_samples": 0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "files": [],
                },
                "val": {
                    "n_samples": n_val_samples,
                    "n_positive": tp + fn,
                    "n_negative": tn + fp,
                    "files": [],
                },
                "features": {
                    "n_features": len(feature_names),
                    "feature_names": feature_names,
                },
                "feature_family": "rf_v1",
                "rf_threshold": rf_threshold,
                "feature_config": feature_config,
                "hyperparameters": params,
                "metrics": report_metrics,
                "model_path": str(rf_model_path),
            }
            write_json(report_path, report)

            self.run_manager.update_outputs(
                run_state.run_id,
                rf_model_path=str(rf_model_path),
                rf_training_report_path=str(report_path),
            )
            return self.run_manager.mark_completed(run_state.run_id)
        except Exception as exc:
            self.run_manager.mark_failed(run_state.run_id, f"{type(exc).__name__}: {exc}")
            raise

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

        elif run_state.run_type == "rf_training":
            from alpaca_pipelines.rf_training.executor import execute_rf_training

            return execute_rf_training(run_state, self.environment, self.run_manager)
        elif run_state.run_type == "rf_inference":
            from alpaca_pipelines.rf_inference.executor import execute_rf_inference

            return execute_rf_inference(run_state, self.environment, self.run_manager)

        raise ValueError("Unknown run type: {}".format(run_state.run_type))

    # ------------------------------------------------------------------
    # Run status and inspection
    # ------------------------------------------------------------------

    def get_run_status(self, run_id: str) -> RunState:
        """Poll the current status of a run."""
        return self.run_manager.find_run(run_id)

    def rename_run(self, run_id: str, new_run_name: str) -> RunState:
        """Rename a run display name in persisted run spec."""
        return self.run_manager.rename_run(run_id, new_run_name)

    def list_runs(
        self,
        run_type: RunType | None = None,
        status_filter: RunStatus | None = None,
    ) -> list[RunState]:
        """List all runs, optionally filtered."""
        return self.run_manager.list_runs(run_type=run_type, status_filter=status_filter)

    def cancel_run(self, run_id: str) -> RunState:
        """Cancel a run from the created or submitted state."""
        run_state = self.run_manager.find_run(run_id)
        if run_state.status == "created":
            return self.run_manager.mark_cancelled(run_id)
        if run_state.status == "submitted":
            if run_state.slurm_job_id is None:
                raise ValueError(
                    "Cannot cancel submitted run without slurm_job_id: {}".format(run_id)
                )
            self._cancel_slurm_job(run_state.slurm_job_id)
            return self.run_manager.mark_cancelled(run_id)
        raise ValueError(
            "Cannot cancel run {}: status is {} (expected created or submitted)".format(
                run_id, run_state.status
            )
        )

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

    def submit_run(
        self,
        run_id: str,
        slurm_config: SlurmConfig | None = None,
    ) -> RunState:
        """Generate and submit a SLURM job for a created run."""
        run_state = self.run_manager.find_run(run_id)
        if run_state.status != "created":
            raise ValueError(
                "Cannot submit run {}: status is {} (expected created)".format(
                    run_id, run_state.status
                )
            )
        if run_state.submitted_at is not None or run_state.slurm_job_id is not None:
            raise ValueError("Run already has submission metadata: {}".format(run_id))

        script_path = self.generate_slurm_script(run_id=run_id, slurm_config=slurm_config)
        slurm_job_id = self._submit_slurm_job(script_path)
        return self.run_manager.mark_submitted(run_id, slurm_job_id)

    def migrate_backend_meta(
        self,
        runs_root: Path | None = None,
    ) -> MigrationSummary:
        """Backfill legacy backend_meta.json fields into run_state.json."""
        target_root = runs_root if runs_root is not None else self.environment.runs_root
        return migrate_backend_meta(target_root, self.run_manager)

    # ------------------------------------------------------------------
    # Collection standardization / dataset workflows
    # ------------------------------------------------------------------

    def start_standardizer_scan(self) -> dict[str, Any]:
        record = self.workflow_ops.start(
            workflow="standardizer",
            kind="scan",
            spec={},
            artifact_name="scan_report.json",
        )
        return record.model_dump()

    def start_standardizer_import(self, identity_map_path: str) -> dict[str, Any]:
        self.workflow_ops.ensure_no_active("standardizer", "import")
        record = self.workflow_ops.start(
            workflow="standardizer",
            kind="import",
            spec={"identity_map_path": identity_map_path},
        )
        return record.model_dump()

    def start_standardizer_plan(self, identity_map_path: str) -> dict[str, Any]:
        self.workflow_ops.ensure_no_active("standardizer", "plan")
        record = self.workflow_ops.start(
            workflow="standardizer",
            kind="plan",
            spec={"identity_map_path": identity_map_path},
            artifact_name="plan.json",
        )
        return record.model_dump()

    def start_standardizer_apply(
        self,
        *,
        plan_job_id: str,
        confirmation_phrase: str,
    ) -> dict[str, Any]:
        self.workflow_ops.ensure_no_active("standardizer", "apply")
        record = self.workflow_ops.start(
            workflow="standardizer",
            kind="apply",
            spec={
                "plan_job_id": plan_job_id,
                "confirmation_phrase": confirmation_phrase,
            },
            artifact_name="apply_result.json",
            rollback_artifact_name="rollback_artifact.json",
        )
        return record.model_dump()

    def start_standardizer_index(
        self,
        *,
        identity_map_path: str,
        min_quality: int,
    ) -> dict[str, Any]:
        self.workflow_ops.ensure_no_active("standardizer", "index")
        record = self.workflow_ops.start(
            workflow="standardizer",
            kind="index",
            spec={
                "identity_map_path": identity_map_path,
                "min_quality": min_quality,
            },
        )
        return record.model_dump()

    def get_workflow_operation(self, job_id: str) -> dict[str, Any]:
        return self.workflow_ops.get(job_id).model_dump()

    def fail_workflow_operation(
        self,
        *,
        job_id: str,
        error: str,
        error_kind: str,
    ) -> dict[str, Any]:
        return self.workflow_ops.fail(
            job_id,
            error=error,
            error_kind=error_kind,
        ).model_dump()

    def delete_failed_workflow_operation(self, *, job_id: str) -> dict[str, Any]:
        operation = self.workflow_ops.delete_failed(job_id)
        return {
            "job_id": operation.job_id,
            "workflow": operation.workflow,
            "kind": operation.kind,
            "status": operation.status,
            "job_dir": operation.job_dir,
            "deleted": True,
        }

    def get_standardizer_status(self) -> dict[str, Any]:
        return {
            "last_import": self._dump_latest_operation("standardizer", "import"),
            "last_scan": self._dump_latest_operation("standardizer", "scan"),
            "last_plan": self._dump_latest_operation("standardizer", "plan"),
            "last_apply": self._dump_latest_operation("standardizer", "apply"),
            "last_index": self._dump_latest_operation("standardizer", "index"),
            "active_jobs": [
                record.model_dump()
                for record in self.workflow_ops.list("standardizer")
                if record.status in {"pending", "running"}
            ],
        }

    def start_dataset_build(
        self,
        *,
        strategy_name: str,
        strategy_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.workflow_ops.ensure_no_active(
            "dataset_builder",
            "build",
            metadata_key="strategy_name",
            metadata_value=strategy_name,
        )
        record = self.workflow_ops.start(
            workflow="dataset_builder",
            kind="build",
            spec={
                "strategy_name": strategy_name,
                "strategy_config": strategy_config,
            },
            metadata={"strategy_name": strategy_name},
        )
        return record.model_dump()

    def start_prepare_review(self, dataset_name: str) -> dict[str, Any]:
        self._ensure_no_active_review(dataset_name)
        record = self.workflow_ops.start(
            workflow="dataset_builder",
            kind="prepare_review",
            spec={"dataset_name": dataset_name},
            metadata={"dataset_name": dataset_name},
        )
        return record.model_dump()

    def start_apply_review(
        self,
        dataset_name: str,
        target_review_table_path: str,
        noise_review_table_path: str,
    ) -> dict[str, Any]:
        self._ensure_no_active_review(dataset_name)
        record = self.workflow_ops.start(
            workflow="dataset_builder",
            kind="apply_review",
            spec={
                "dataset_name": dataset_name,
                "target_review_table_path": target_review_table_path,
                "noise_review_table_path": noise_review_table_path,
            },
            metadata={"dataset_name": dataset_name},
        )
        return record.model_dump()

    def get_dataset_builder_status(self) -> dict[str, Any]:
        return {
            "last_build": self._dump_latest_operation("dataset_builder", "build"),
            "last_prepare_review": self._dump_latest_operation("dataset_builder", "prepare_review"),
            "last_apply_review": self._dump_latest_operation("dataset_builder", "apply_review"),
            "active_jobs": [
                record.model_dump()
                for record in self.workflow_ops.list("dataset_builder")
                if record.status in {"pending", "running"}
            ],
        }

    def execute_workflow_operation(self, job_dir: str) -> None:
        self.workflow_ops.run_worker(Path(job_dir))

    def _dump_latest_operation(self, workflow: str, kind: str) -> dict[str, Any] | None:
        record = self.workflow_ops.latest(workflow, kind)
        return record.model_dump() if record is not None else None

    def _ensure_no_active_review(self, dataset_name: str) -> None:
        for kind in ("prepare_review", "apply_review"):
            self.workflow_ops.ensure_no_active(
                "dataset_builder",
                kind,
                metadata_key="dataset_name",
                metadata_value=dataset_name,
            )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def generate_prediction_review_preview(
        self,
        *,
        manifest_path: Path,
        item_id: str,
        spectrogram_config: PredictionReviewSpectrogramConfig | None = None,
    ) -> dict[str, Any]:
        from alpaca_pipelines.prediction.review.executor import (
            generate_prediction_review_preview,
        )

        return generate_prediction_review_preview(
            run_manager=self.run_manager,
            manifest_path=manifest_path,
            item_id=item_id,
            spectrogram_config=spectrogram_config,
        )

    def generate_prediction_review_batch(
        self,
        *,
        manifest_path: Path,
        spectrogram_config: PredictionReviewSpectrogramConfig | None = None,
    ) -> dict[str, Any]:
        from alpaca_pipelines.prediction.review.executor import (
            generate_prediction_review_batch,
        )

        return generate_prediction_review_batch(
            run_manager=self.run_manager,
            manifest_path=manifest_path,
            spectrogram_config=spectrogram_config,
        )

    def concatenate_prediction_review_clips(
        self,
        *,
        manifest_path: Path,
        output_wav: Path | None = None,
    ) -> dict[str, Any]:
        from alpaca_pipelines.prediction.review.executor import (
            concatenate_prediction_review_clips,
        )

        return concatenate_prediction_review_clips(
            run_manager=self.run_manager,
            manifest_path=manifest_path,
            output_wav=output_wav,
        )

    def export_prediction_review_artifacts(
        self,
        *,
        manifest_path: Path,
        destination_dir: Path,
        item_id: str | None = None,
    ) -> dict[str, Any]:
        from alpaca_pipelines.prediction.review.executor import (
            export_prediction_review_artifacts,
        )

        return export_prediction_review_artifacts(
            run_manager=self.run_manager,
            manifest_path=manifest_path,
            destination_dir=destination_dir,
            item_id=item_id,
        )

    def export_prediction_review_flat_snippets_bundle(
        self,
        *,
        manifest_path: Path,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        from alpaca_pipelines.prediction.review.executor import (
            export_prediction_review_flat_snippets_bundle,
        )

        return export_prediction_review_flat_snippets_bundle(
            run_manager=self.run_manager,
            manifest_path=manifest_path,
            output_dir=output_dir,
        )

    def materialize_curated_prediction_examples(
        self,
        *,
        manifest_path: Path | None = None,
        labels_path: Path | None = None,
        curated_export_manifest: Path | None = None,
        destination_root: Path | None = None,
    ) -> dict[str, Any]:
        return materialize_curated_prediction_examples(
            run_manager=self.run_manager,
            collection_root=self.environment.collection_root,
            datasets_root=self.environment.datasets_root,
            manifest_path=manifest_path,
            labels_path=labels_path,
            curated_export_manifest=curated_export_manifest,
            destination_root=destination_root,
        )

    def list_curated_prediction_categories(
        self,
        *,
        destination_root: Path | None = None,
    ) -> dict[str, Any]:
        return list_curated_prediction_categories(
            collection_root=self.environment.collection_root,
            destination_root=destination_root,
        )

    def list_curated_prediction_sources(
        self,
        *,
        destination_root: Path | None = None,
    ) -> dict[str, Any]:
        return self.list_curated_prediction_categories(destination_root=destination_root)

    def migrate_legacy_curated_prediction_sources(
        self,
        *,
        destination_root: Path | None = None,
        remove_legacy_root: bool = False,
    ) -> dict[str, Any]:
        return migrate_legacy_curated_prediction_sources(
            collection_root=self.environment.collection_root,
            datasets_root=self.environment.datasets_root,
            destination_root=destination_root,
            remove_legacy_root=remove_legacy_root,
        )

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

    def export_prediction_run_selection_tables(
        self,
        prediction_run_id: str,
        freq_low_hz: int = 0,
        freq_high_hz: int = 4000,
        use_rf_filtered: bool = False,
    ) -> dict[str, Any]:
        """Export Raven selection tables for all files in a completed prediction run.

        Writes tables and a summary JSON under the run's persisted selection_tables
        directory and updates run_state.outputs to point at the summary.
        """
        from alpaca_pipelines.postprocessing.executor import export_prediction_run_selection_tables

        run_state = self.run_manager.find_run(prediction_run_id)
        if run_state.run_type != "prediction":
            raise ValueError(
                "export_prediction_run_selection_tables requires a prediction run, got: {}".format(
                    run_state.run_type
                )
            )
        if run_state.status != "completed":
            raise ValueError(
                "Prediction run must be completed to export selection tables, status is: {}".format(
                    run_state.status
                )
            )

        if run_state.outputs.predictions_dir is None:
            raise ValueError("Run outputs missing predictions_dir: {}".format(prediction_run_id))
        predictions_dir = Path(run_state.outputs.predictions_dir)

        if run_state.outputs.prediction_selection_tables_dir is None:
            raise ValueError(
                "Run outputs missing prediction_selection_tables_dir: {}".format(prediction_run_id)
            )
        selection_tables_dir = Path(run_state.outputs.prediction_selection_tables_dir)

        selection_tables_summary = export_prediction_run_selection_tables(
            predictions_dir=predictions_dir,
            selection_tables_dir=selection_tables_dir,
            freq_low_hz=freq_low_hz,
            freq_high_hz=freq_high_hz,
            use_rf_filtered=use_rf_filtered,
        )

        summary_path_value = selection_tables_summary.get("selection_tables_dir")
        if not isinstance(summary_path_value, str) or not summary_path_value:
            raise ValueError(
                "Invalid selection table summary payload (missing selection_tables_dir)"
            )

        summary_file_path = selection_tables_dir / "selection_tables_summary.json"
        self.run_manager.update_outputs(
            run_state.run_id,
            prediction_selection_tables_summary_path=str(summary_file_path),
        )

        return selection_tables_summary

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

    def _submit_slurm_job(self, script_path: Path) -> str:
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sbatch failed")
        raw = result.stdout.strip()
        job_id = raw.split(";", 1)[0].strip()
        if not job_id.isdigit():
            raise ValueError("Could not parse job ID from sbatch output: {!r}".format(raw))
        return job_id

    def _cancel_slurm_job(self, job_id: str) -> None:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "scancel failed for {}".format(job_id))
