"""
CLI entry point for alpaca-pipelines.

This is a thin wrapper around ``PipelineAPI``. All logic lives in the
API layer; the CLI only handles argument parsing and console output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

from alpaca_pipelines.api import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.prediction.review import PredictionReviewSpectrogramConfig
from alpaca_pipelines.rf_training.config import RfTrainingRunSpec
from alpaca_pipelines.slurm.config import SlurmConfig
from alpaca_pipelines.training.config import TrainingRunSpec


def _load_json_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        print("Config file not found: {}".format(config_path), file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        print("Expected JSON object in config: {}".format(config_path), file=sys.stderr)
        sys.exit(1)
    return data


def _get_api() -> PipelineAPI:
    try:
        environment = PipelineEnvironment.from_env()
        return PipelineAPI(environment)
    except (EnvironmentError, FileNotFoundError) as exc:
        print("Environment error: {}".format(exc), file=sys.stderr)
        sys.exit(1)


def _emit_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


def _format_error_message(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if message:
        return message
    return exc.__class__.__name__


def _exit_with_error(exc: BaseException) -> NoReturn:
    print(_format_error_message(exc), file=sys.stderr)
    sys.exit(1)


def _run_state_payload(run_state: RunState) -> dict[str, Any]:
    return run_state.model_dump()


def _load_review_spectrogram_config(
    config_path: str | None,
) -> PredictionReviewSpectrogramConfig | None:
    if config_path is None:
        return None
    config_data = _load_json_config(config_path)
    return PredictionReviewSpectrogramConfig.model_validate(config_data)


def _cmd_create(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        config_data = _load_json_config(args.config)

        if args.run_type == "training":
            run_state = api.create_training_run(TrainingRunSpec.model_validate(config_data))
        elif args.run_type == "rf_training":
            run_state = api.create_rf_training_run(RfTrainingRunSpec.model_validate(config_data))
        elif args.run_type == "prediction":
            run_state = api.create_prediction_run(PredictionRunSpec.model_validate(config_data))
        elif args.run_type == "evaluation":
            run_state = api.create_evaluation_run(EvaluationRunSpec.model_validate(config_data))
        else:
            raise ValueError("Unknown run type: {}".format(args.run_type))
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(_run_state_payload(run_state))
        return

    print("Created {} run: {}".format(run_state.run_type, run_state.run_id))
    print("  Run dir: {}".format(run_state.run_dir))


def _cmd_execute(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        run_state = api.execute_run(args.run_id)
        print("Run {} completed with status: {}".format(run_state.run_id, run_state.status))
    except Exception as exc:
        _exit_with_error(RuntimeError("Run failed: {}".format(exc)))


def _cmd_list(args: argparse.Namespace) -> None:
    api = _get_api()
    run_type = getattr(args, "type", None)
    status_filter = getattr(args, "status", None)
    runs = api.list_runs(run_type=run_type, status_filter=status_filter)

    if args.json:
        _emit_json({"runs": [_run_state_payload(run_state) for run_state in runs]})
        return

    if not runs:
        print("No runs found.")
        return

    print("{:<38} {:<12} {:<10} {:<20}".format("RUN ID", "TYPE", "STATUS", "CREATED"))
    print("-" * 82)
    for run in runs:
        print(
            "{:<38} {:<12} {:<10} {:<20}".format(
                run.run_id, run.run_type, run.status, run.created_at
            )
        )


def _cmd_inspect(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        run_state = api.get_run_status(args.run_id)
    except FileNotFoundError as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(_run_state_payload(run_state))
        return

    print("Run:        {}".format(run_state.run_id))
    print("Type:       {}".format(run_state.run_type))
    print("Status:     {}".format(run_state.status))
    print("Created:    {}".format(run_state.created_at))
    if run_state.submitted_at:
        print("Submitted:  {}".format(run_state.submitted_at))
    if run_state.started_at:
        print("Started:    {}".format(run_state.started_at))
    if run_state.completed_at:
        print("Completed:  {}".format(run_state.completed_at))
    if run_state.slurm_job_id:
        print("Slurm job:  {}".format(run_state.slurm_job_id))
    if run_state.error_message:
        print("Error:      {}".format(run_state.error_message))
    print("Run dir:    {}".format(run_state.run_dir))

    if run_state.progress.current_epoch is not None:
        print(
            "Progress:   epoch {}/{}".format(
                run_state.progress.current_epoch,
                run_state.progress.total_epochs,
            )
        )

    outputs = run_state.outputs.model_dump()
    non_null_outputs = {
        key: value for key, value in outputs.items() if value is not None and value is not False
    }
    if non_null_outputs:
        print("Outputs:")
        for key, value in non_null_outputs.items():
            print("  {}: {}".format(key, value))


def _cmd_cancel(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        run_state = api.cancel_run(args.run_id)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(_run_state_payload(run_state))
        return

    print("Cancelled run: {}".format(run_state.run_id))


def _cmd_generate_slurm(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        slurm_config = None
        if args.slurm_config:
            slurm_data = _load_json_config(args.slurm_config)
            slurm_config = SlurmConfig.model_validate(slurm_data)
        script_path = api.generate_slurm_script(
            run_id=args.run_id,
            slurm_config=slurm_config,
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json({"run_id": args.run_id, "script_path": str(script_path)})
        return

    print("SLURM script generated: {}".format(script_path))
    print("Submit with: sbatch {}".format(script_path))


def _cmd_submit(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        slurm_config = None
        if args.slurm_config:
            slurm_data = _load_json_config(args.slurm_config)
            slurm_config = SlurmConfig.model_validate(slurm_data)
        run_state = api.submit_run(args.run_id, slurm_config=slurm_config)
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(_run_state_payload(run_state))
        return

    print("Submitted run: {}".format(run_state.run_id))
    print("  Slurm job ID: {}".format(run_state.slurm_job_id))


def _cmd_export_selection_tables(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        summary = api.export_prediction_run_selection_tables(
            prediction_run_id=args.run_id,
            freq_low_hz=args.freq_low_hz,
            freq_high_hz=args.freq_high_hz,
            use_rf_filtered=args.use_rf_filtered,
        )
    except Exception as exc:
        _exit_with_error(RuntimeError("Export failed: {}".format(exc)))

    if args.json:
        _emit_json(summary)
        return

    selection_tables_dir = summary.get("selection_tables_dir")
    if not isinstance(selection_tables_dir, str):
        _exit_with_error(RuntimeError("Export failed: missing selection_tables_dir"))
    summary_path = str(Path(selection_tables_dir) / "selection_tables_summary.json")
    print("Selection tables exported.")
    print("  Dir:      {}".format(selection_tables_dir))
    print("  Summary:  {}".format(summary_path))

    files = summary.get("files", [])
    if isinstance(files, list) and files:
        print("  Files:")
        for entry in files:
            if isinstance(entry, dict) and "selection_table" in entry:
                print("    {}".format(entry["selection_table"]))


def _cmd_migrate_backend_meta(args: argparse.Namespace) -> None:
    api = _get_api()
    runs_root = Path(args.runs_root) if args.runs_root else None
    try:
        summary = api.migrate_backend_meta(runs_root=runs_root)
    except Exception as exc:
        _exit_with_error(exc)

    payload = summary.to_dict()
    if args.json:
        _emit_json(payload)
        return

    print("Migration complete.")
    print("  Migrated: {}".format(len(summary.migrated)))
    print("  Skipped: {}".format(len(summary.skipped)))


def _cmd_prediction_review_preview(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.generate_prediction_review_preview(
            manifest_path=Path(args.manifest),
            item_id=args.item_id,
            spectrogram_config=_load_review_spectrogram_config(args.spectrogram_config),
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(payload)
        return

    print("Prediction review preview generated.")
    print("  Run:      {}".format(payload["prediction_run_id"]))
    print("  Session:  {}".format(payload["session_id"]))
    print("  Item:     {}".format(payload["item_id"]))
    print("  Summary:  {}".format(payload["summary_path"]))


def _cmd_prediction_review_generate(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.generate_prediction_review_batch(
            manifest_path=Path(args.manifest),
            spectrogram_config=_load_review_spectrogram_config(args.spectrogram_config),
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(payload)
        return

    print("Prediction review batch generation complete.")
    print("  Run:      {}".format(payload["prediction_run_id"]))
    print("  Session:  {}".format(payload["session_id"]))
    print("  Items:    {}".format(payload["n_items"]))
    print("  Summary:  {}".format(payload["summary_path"]))


def _cmd_prediction_review_concat(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.concatenate_prediction_review_clips(
            manifest_path=Path(args.manifest),
            output_wav=Path(args.output_wav) if args.output_wav is not None else None,
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(payload)
        return

    print("Prediction review concat complete.")
    print("  Run:      {}".format(payload["prediction_run_id"]))
    print("  Session:  {}".format(payload["session_id"]))
    print("  Items:    {}".format(payload["n_items"]))
    print("  Concat:   {}".format(payload["concat_wav"]))


def _cmd_prediction_review_export(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.export_prediction_review_artifacts(
            manifest_path=Path(args.manifest),
            destination_dir=Path(args.destination_dir),
            item_id=args.item_id,
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(payload)
        return

    print("Prediction review export complete.")
    print("  Run:          {}".format(payload["prediction_run_id"]))
    print("  Session:      {}".format(payload["session_id"]))
    print("  Destination:  {}".format(payload["destination_dir"]))
    print("  Items:        {}".format(payload["n_items"]))
    print("  Summary:      {}".format(payload["summary_path"]))


def _cmd_prediction_review_materialize_curated(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.materialize_curated_prediction_examples(
            manifest_path=Path(args.manifest),
            labels_path=Path(args.labels),
            destination_root=Path(args.destination_root) if args.destination_root else None,
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(payload)
        return

    print("Curated prediction examples materialized.")
    print("  Curated root: {}".format(payload["curated_source_root"]))
    print("  Run:          {}".format(payload["prediction_run_id"]))
    print("  Session:      {}".format(payload["review_session_id"]))
    print("  Total items:  {}".format(payload["total_items"]))
    print("  Created:      {}".format(payload["created_count"]))
    print("  Updated:      {}".format(payload["updated_count"]))
    print("  Skipped:      {}".format(payload["skipped_count"]))
    print("  Target:       {}".format(payload["counts_by_label"]["target"]))
    print("  Noise:        {}".format(payload["counts_by_label"]["noise"]))
    for manifest_path in payload["manifest_paths"]:
        print("  Manifest:     {}".format(manifest_path))


def _cmd_curated_source_status(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.list_curated_prediction_sources(
            destination_root=Path(args.destination_root) if args.destination_root else None,
        )
    except Exception as exc:
        _exit_with_error(exc)

    if args.json:
        _emit_json(payload)
        return

    print("Curated source status")
    print("  Curated root: {}".format(payload["curated_source_root"]))
    print("  Manifests:    {}".format(len(payload["manifests"])))
    print("  By label:     {}".format(payload["counts_by_label"]))
    print("  By collection: {}".format(payload["counts_by_collection"]))
    if payload["warnings"]:
        print("  Warnings:")
        for warning in payload["warnings"]:
            print("    {}".format(warning))


def _cmd_standardizer_scan(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_standardizer_scan()
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_standardizer_import(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_standardizer_import(args.identity_map)
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_standardizer_plan(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_standardizer_plan(args.identity_map)
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_standardizer_apply(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_standardizer_apply(
            plan_job_id=args.plan_job_id,
            confirmation_phrase=args.confirmation_phrase,
        )
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_standardizer_index(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_standardizer_index(
            identity_map_path=args.identity_map,
            min_quality=args.min_source_quality_to_keep,
        )
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_standardizer_job(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.get_workflow_operation(args.job_id)
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_standardizer_status(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.get_standardizer_status()
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_dataset_build(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        config_data = _load_json_config(args.config)
        payload = api.start_dataset_build(
            strategy_name=args.strategy_name,
            strategy_config=config_data,
        )
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_dataset_prepare_review(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_prepare_review(args.dataset_name)
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_dataset_apply_review(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.start_apply_review(
            args.dataset_name,
            args.target_review_table,
            args.noise_review_table,
        )
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_dataset_job(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.get_workflow_operation(args.job_id)
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_dataset_status(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.get_dataset_builder_status()
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_execute_operation(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        api.execute_workflow_operation(args.job_dir)
    except Exception as exc:
        _exit_with_error(exc)


def _cmd_fail_operation(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.fail_workflow_operation(
            job_id=args.job_id,
            error=args.error,
            error_kind=args.error_kind,
        )
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _cmd_delete_failed_operation(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        payload = api.delete_failed_workflow_operation(job_id=args.job_id)
    except Exception as exc:
        _exit_with_error(exc)
    _emit_json(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpaca-pipelines",
        description="Mid-level orchestrator for bioacoustics DL pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    create_parser = subparsers.add_parser("create", help="Create a new pipeline run")
    create_parser.add_argument(
        "run_type",
        choices=["training", "rf_training", "prediction", "evaluation"],
        help="Type of run to create",
    )
    create_parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON configuration file for the run",
    )
    create_parser.add_argument("--json", action="store_true")

    execute_parser = subparsers.add_parser("execute", help="Execute a pipeline run")
    execute_parser.add_argument("--run-id", required=True, help="Run ID to execute")

    list_parser = subparsers.add_parser("list", help="List pipeline runs")
    list_parser.add_argument(
        "--type",
        choices=["training", "rf_training", "prediction", "evaluation"],
        default=None,
        help="Filter by run type",
    )
    list_parser.add_argument(
        "--status",
        choices=["created", "submitted", "running", "completed", "failed", "cancelled"],
        default=None,
        help="Filter by status",
    )
    list_parser.add_argument("--json", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a pipeline run")
    inspect_parser.add_argument("--run-id", required=True, help="Run ID to inspect")
    inspect_parser.add_argument("--json", action="store_true")

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pipeline run")
    cancel_parser.add_argument("--run-id", required=True, help="Run ID to cancel")
    cancel_parser.add_argument("--json", action="store_true")

    slurm_parser = subparsers.add_parser("generate-slurm", help="Generate a SLURM batch script")
    slurm_parser.add_argument("--run-id", required=True, help="Run ID")
    slurm_parser.add_argument(
        "--slurm-config",
        default=None,
        help="Path to SLURM config JSON (optional)",
    )
    slurm_parser.add_argument("--json", action="store_true")

    submit_parser = subparsers.add_parser("submit", help="Submit a created pipeline run")
    submit_parser.add_argument("--run-id", required=True, help="Run ID")
    submit_parser.add_argument(
        "--slurm-config",
        default=None,
        help="Path to SLURM config JSON (optional)",
    )
    submit_parser.add_argument("--json", action="store_true")

    export_tables_parser = subparsers.add_parser(
        "export-selection-tables",
        help="Export Raven selection tables for a completed prediction run",
    )
    export_tables_parser.add_argument("--run-id", required=True, help="Prediction run ID")
    export_tables_parser.add_argument("--freq-low-hz", type=int, default=0)
    export_tables_parser.add_argument("--freq-high-hz", type=int, default=4000)
    export_tables_parser.add_argument("--use-rf-filtered", action="store_true")
    export_tables_parser.add_argument("--json", action="store_true")

    migrate_parser = subparsers.add_parser(
        "migrate-backend-meta",
        help="Backfill legacy backend_meta.json fields into run_state.json",
    )
    migrate_parser.add_argument("--runs-root", default=None)
    migrate_parser.add_argument("--json", action="store_true")

    review_preview_parser = subparsers.add_parser("prediction-review-preview")
    review_preview_parser.add_argument("--manifest", required=True)
    review_preview_parser.add_argument("--item-id", required=True)
    review_preview_parser.add_argument("--spectrogram-config", default=None)
    review_preview_parser.add_argument("--json", action="store_true")

    review_generate_parser = subparsers.add_parser("prediction-review-generate")
    review_generate_parser.add_argument("--manifest", required=True)
    review_generate_parser.add_argument("--spectrogram-config", default=None)
    review_generate_parser.add_argument("--json", action="store_true")

    review_concat_parser = subparsers.add_parser("prediction-review-concat")
    review_concat_parser.add_argument("--manifest", required=True)
    review_concat_parser.add_argument("--output-wav", default=None)
    review_concat_parser.add_argument("--json", action="store_true")

    review_export_parser = subparsers.add_parser("prediction-review-export")
    review_export_parser.add_argument("--manifest", required=True)
    review_export_parser.add_argument("--destination-dir", required=True)
    review_export_parser.add_argument("--item-id", default=None)
    review_export_parser.add_argument("--json", action="store_true")

    review_materialize_parser = subparsers.add_parser("prediction-review-materialize-curated")
    review_materialize_parser.add_argument("--manifest", required=True)
    review_materialize_parser.add_argument("--labels", required=True)
    review_materialize_parser.add_argument("--destination-root", default=None)
    review_materialize_parser.add_argument("--json", action="store_true")

    curated_status_parser = subparsers.add_parser("curated-source-status")
    curated_status_parser.add_argument("--destination-root", default=None)
    curated_status_parser.add_argument("--json", action="store_true")

    standardizer_scan_parser = subparsers.add_parser("standardizer-scan")
    standardizer_scan_parser.add_argument("--json", action="store_true")

    standardizer_import_parser = subparsers.add_parser("standardizer-import")
    standardizer_import_parser.add_argument("--identity-map", required=True)
    standardizer_import_parser.add_argument("--json", action="store_true")

    standardizer_plan_parser = subparsers.add_parser("standardizer-plan")
    standardizer_plan_parser.add_argument("--identity-map", required=True)
    standardizer_plan_parser.add_argument("--json", action="store_true")

    standardizer_apply_parser = subparsers.add_parser("standardizer-apply")
    standardizer_apply_parser.add_argument("--plan-job-id", required=True)
    standardizer_apply_parser.add_argument("--confirmation-phrase", required=True)
    standardizer_apply_parser.add_argument("--json", action="store_true")

    standardizer_index_parser = subparsers.add_parser("standardizer-index")
    standardizer_index_parser.add_argument("--identity-map", required=True)
    standardizer_index_parser.add_argument(
        "--min-source-quality-to-keep",
        type=int,
        default=1,
    )
    standardizer_index_parser.add_argument("--json", action="store_true")

    standardizer_job_parser = subparsers.add_parser("standardizer-job")
    standardizer_job_parser.add_argument("--job-id", required=True)
    standardizer_job_parser.add_argument("--json", action="store_true")

    standardizer_status_parser = subparsers.add_parser("standardizer-status")
    standardizer_status_parser.add_argument("--json", action="store_true")

    dataset_build_parser = subparsers.add_parser("dataset-build")
    dataset_build_parser.add_argument("--strategy-name", required=True)
    dataset_build_parser.add_argument("--config", required=True)
    dataset_build_parser.add_argument("--json", action="store_true")

    dataset_prepare_parser = subparsers.add_parser("dataset-prepare-review")
    dataset_prepare_parser.add_argument("--dataset-name", required=True)
    dataset_prepare_parser.add_argument("--json", action="store_true")

    dataset_apply_parser = subparsers.add_parser("dataset-apply-review")
    dataset_apply_parser.add_argument("--dataset-name", required=True)
    dataset_apply_parser.add_argument("--target-review-table", required=True)
    dataset_apply_parser.add_argument("--noise-review-table", required=True)
    dataset_apply_parser.add_argument("--json", action="store_true")

    dataset_job_parser = subparsers.add_parser("dataset-job")
    dataset_job_parser.add_argument("--job-id", required=True)
    dataset_job_parser.add_argument("--json", action="store_true")

    dataset_status_parser = subparsers.add_parser("dataset-status")
    dataset_status_parser.add_argument("--json", action="store_true")

    execute_operation_parser = subparsers.add_parser("_execute-operation")
    execute_operation_parser.add_argument("--job-dir", required=True)

    fail_operation_parser = subparsers.add_parser("fail-operation")
    fail_operation_parser.add_argument("--job-id", required=True)
    fail_operation_parser.add_argument("--error-kind", required=True)
    fail_operation_parser.add_argument("--error", required=True)
    fail_operation_parser.add_argument("--json", action="store_true")

    delete_failed_operation_parser = subparsers.add_parser("delete-failed-operation")
    delete_failed_operation_parser.add_argument("--job-id", required=True)
    delete_failed_operation_parser.add_argument("--json", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    command_handlers = {
        "create": _cmd_create,
        "execute": _cmd_execute,
        "list": _cmd_list,
        "inspect": _cmd_inspect,
        "cancel": _cmd_cancel,
        "generate-slurm": _cmd_generate_slurm,
        "submit": _cmd_submit,
        "export-selection-tables": _cmd_export_selection_tables,
        "migrate-backend-meta": _cmd_migrate_backend_meta,
        "prediction-review-preview": _cmd_prediction_review_preview,
        "prediction-review-generate": _cmd_prediction_review_generate,
        "prediction-review-concat": _cmd_prediction_review_concat,
        "prediction-review-export": _cmd_prediction_review_export,
        "prediction-review-materialize-curated": _cmd_prediction_review_materialize_curated,
        "curated-source-status": _cmd_curated_source_status,
        "standardizer-scan": _cmd_standardizer_scan,
        "standardizer-import": _cmd_standardizer_import,
        "standardizer-plan": _cmd_standardizer_plan,
        "standardizer-apply": _cmd_standardizer_apply,
        "standardizer-index": _cmd_standardizer_index,
        "standardizer-job": _cmd_standardizer_job,
        "standardizer-status": _cmd_standardizer_status,
        "dataset-build": _cmd_dataset_build,
        "dataset-prepare-review": _cmd_dataset_prepare_review,
        "dataset-apply-review": _cmd_dataset_apply_review,
        "dataset-job": _cmd_dataset_job,
        "dataset-status": _cmd_dataset_status,
        "_execute-operation": _cmd_execute_operation,
        "fail-operation": _cmd_fail_operation,
        "delete-failed-operation": _cmd_delete_failed_operation,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
