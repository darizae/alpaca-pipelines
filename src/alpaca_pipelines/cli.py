"""
CLI entry point for alpaca-pipelines.

This is a thin wrapper around ``PipelineAPI``.  All logic lives in the
API layer; the CLI only handles argument parsing and console output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alpaca_pipelines.api import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.rf_training.config import RfTrainingRunSpec
from alpaca_pipelines.slurm.config import SlurmConfig
from alpaca_pipelines.training.config import TrainingRunSpec


def _load_json_config(config_path: str) -> dict:  # type: ignore[type-arg]
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


def _cmd_create(args: argparse.Namespace) -> None:
    api = _get_api()
    config_data = _load_json_config(args.config)

    if args.run_type == "training":
        spec = TrainingRunSpec.model_validate(config_data)
        run_state = api.create_training_run(spec)
    elif args.run_type == "rf_training":
        spec = RfTrainingRunSpec.model_validate(config_data)  # type: ignore[assignment]
        run_state = api.create_rf_training_run(spec)  # type: ignore[arg-type]
    elif args.run_type == "prediction":
        spec = PredictionRunSpec.model_validate(config_data)  # type: ignore[assignment]
        run_state = api.create_prediction_run(spec)  # type: ignore[arg-type]
    elif args.run_type == "evaluation":
        spec = EvaluationRunSpec.model_validate(config_data)  # type: ignore[assignment]
        run_state = api.create_evaluation_run(spec)  # type: ignore[arg-type]
    else:
        print("Unknown run type: {}".format(args.run_type), file=sys.stderr)
        sys.exit(1)

    print("Created {} run: {}".format(run_state.run_type, run_state.run_id))
    print("  Run dir: {}".format(run_state.run_dir))


def _cmd_execute(args: argparse.Namespace) -> None:
    api = _get_api()
    try:
        run_state = api.execute_run(args.run_id)
        print("Run {} completed with status: {}".format(run_state.run_id, run_state.status))
    except Exception as exc:
        print("Run failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    api = _get_api()
    run_type = getattr(args, "type", None)
    status_filter = getattr(args, "status", None)
    runs = api.list_runs(run_type=run_type, status_filter=status_filter)

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
    except FileNotFoundError:
        print("Run not found: {}".format(args.run_id), file=sys.stderr)
        sys.exit(1)

    print("Run:        {}".format(run_state.run_id))
    print("Type:       {}".format(run_state.run_type))
    print("Status:     {}".format(run_state.status))
    print("Created:    {}".format(run_state.created_at))
    if run_state.started_at:
        print("Started:    {}".format(run_state.started_at))
    if run_state.completed_at:
        print("Completed:  {}".format(run_state.completed_at))
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
        print("Cancelled run: {}".format(run_state.run_id))
    except (FileNotFoundError, ValueError) as exc:
        print("Cannot cancel: {}".format(exc), file=sys.stderr)
        sys.exit(1)


def _cmd_generate_slurm(args: argparse.Namespace) -> None:
    api = _get_api()
    slurm_config = None
    if hasattr(args, "slurm_config") and args.slurm_config:
        slurm_data = _load_json_config(args.slurm_config)
        slurm_config = SlurmConfig.model_validate(slurm_data)

    try:
        script_path = api.generate_slurm_script(
            run_id=args.run_id,
            slurm_config=slurm_config,
        )
        print("SLURM script generated: {}".format(script_path))
        print("Submit with: sbatch {}".format(script_path))
    except FileNotFoundError as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        sys.exit(1)


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

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a pipeline run")
    inspect_parser.add_argument("--run-id", required=True, help="Run ID to inspect")

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pipeline run")
    cancel_parser.add_argument("--run-id", required=True, help="Run ID to cancel")

    slurm_parser = subparsers.add_parser("generate-slurm", help="Generate a SLURM batch script")
    slurm_parser.add_argument("--run-id", required=True, help="Run ID")
    slurm_parser.add_argument(
        "--slurm-config",
        default=None,
        help="Path to SLURM config JSON (optional)",
    )

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
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)
