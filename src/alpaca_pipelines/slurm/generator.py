"""
SLURM batch script generator.

Generates .sbatch scripts that execute pipeline runs via the CLI.
The generated script handles environment setup, module loading,
and delegates to ``alpaca-pipelines execute --run-id <id>``.
"""

from __future__ import annotations

from pathlib import Path

from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.slurm.config import SlurmConfig


def _build_sbatch_header(
    slurm_config: SlurmConfig,
    run_state: RunState,
) -> list[str]:
    """Build the #SBATCH directives section."""
    run_dir = Path(run_state.run_dir)
    job_name = slurm_config.job_name or "{}_{}".format(
        run_state.run_type, run_state.run_id[:8]
    )

    lines: list[str] = [
        "#!/bin/bash",
        "#SBATCH --job-name={}".format(job_name),
        "#SBATCH --partition={}".format(slurm_config.partition),
        "#SBATCH --nodes={}".format(slurm_config.nodes),
        "#SBATCH --ntasks={}".format(slurm_config.ntasks),
        "#SBATCH --cpus-per-task={}".format(slurm_config.cpus_per_task),
        "#SBATCH --mem={}G".format(slurm_config.memory_gb),
        "#SBATCH --time={}".format(slurm_config.time_limit),
        "#SBATCH --output={}".format(run_dir / "slurm" / slurm_config.output_pattern),
        "#SBATCH --error={}".format(run_dir / "slurm" / slurm_config.error_pattern),
    ]

    if slurm_config.gpus > 0:
        if slurm_config.gpu_type:
            lines.append(
                "#SBATCH --gres=gpu:{}:{}".format(slurm_config.gpu_type, slurm_config.gpus)
            )
        else:
            lines.append("#SBATCH --gres=gpu:{}".format(slurm_config.gpus))

    if slurm_config.account:
        lines.append("#SBATCH --account={}".format(slurm_config.account))

    for extra_line in slurm_config.extra_sbatch_lines:
        lines.append("#SBATCH {}".format(extra_line))

    return lines


def _build_environment_section(
    slurm_config: SlurmConfig,
    environment_vars: dict[str, str],
) -> list[str]:
    """Build the environment setup section."""
    lines: list[str] = ["", "# --- Environment setup ---"]

    if slurm_config.modules:
        for module_name in slurm_config.modules:
            lines.append("module load {}".format(module_name))
        lines.append("")

    if slurm_config.conda_env:
        lines.append("conda activate {}".format(slurm_config.conda_env))
        lines.append("")
    elif slurm_config.venv_path:
        lines.append("source {}/bin/activate".format(slurm_config.venv_path))
        lines.append("")

    lines.append("# --- Pipeline environment ---")
    for key, value in sorted(environment_vars.items()):
        lines.append('export {}="{}"'.format(key, value))

    for key, value in sorted(slurm_config.extra_env_vars.items()):
        lines.append('export {}="{}"'.format(key, value))

    return lines


def _build_execution_section(run_state: RunState) -> list[str]:
    """Build the actual execution command."""
    return [
        "",
        "# --- Execute pipeline run ---",
        'echo "Starting run: {}"'.format(run_state.run_id),
        'echo "Run type: {}"'.format(run_state.run_type),
        'echo "Hostname: $(hostname)"',
        'echo "GPU info:"',
        "nvidia-smi 2>/dev/null || echo 'No GPU available'",
        "",
        "alpaca-pipelines execute --run-id {}".format(run_state.run_id),
        "",
        "EXIT_CODE=$?",
        'echo "Run completed with exit code: $EXIT_CODE"',
        "exit $EXIT_CODE",
    ]


def generate_slurm_script(
    run_state: RunState,
    slurm_config: SlurmConfig,
    environment_vars: dict[str, str],
) -> Path:
    """Generate a SLURM batch script for executing a pipeline run.

    Writes the script to the run's slurm/ directory and returns the path.
    """
    run_dir = Path(run_state.run_dir)
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    script_path = slurm_dir / "job.sbatch"

    lines: list[str] = []
    lines.extend(_build_sbatch_header(slurm_config, run_state))
    lines.extend(_build_environment_section(slurm_config, environment_vars))
    lines.extend(_build_execution_section(run_state))

    script_content = "\n".join(lines) + "\n"
    script_path.write_text(script_content, encoding="utf-8")

    return script_path
