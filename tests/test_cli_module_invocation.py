from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from alpaca_pipelines.io_utils import write_json


def _build_cli_env(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir()
    datasets_root.mkdir()
    runs_root.mkdir()
    (datasets_root / "dataset-a").mkdir()
    write_json(collection_root / "merged_index.json", {"meta": {}, "entries": []})
    tape_path = collection_root / "audio_collection_alpha" / "raw_recordings"
    tape_path.mkdir(parents=True)
    (tape_path / "tape_001.wav").write_bytes(b"")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo_root / "src"),
            "ALPACA_COLLECTION_ROOT": str(collection_root),
            "ALPACA_MERGED_INDEX": str(collection_root / "merged_index.json"),
            "ALPACA_DATASETS_ROOT": str(datasets_root),
            "ALPACA_RUNS_ROOT": str(runs_root),
        }
    )
    return repo_root, env


def test_python_module_invocation_executes_cli(tmp_path: Path) -> None:
    repo_root, env = _build_cli_env(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "alpaca_pipelines.cli", "dataset-status", "--json"],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {
        "last_build": None,
        "last_prepare_review": None,
        "last_apply_review": None,
        "active_jobs": [],
    }


def test_create_json_outputs_parseable_payload_for_each_run_type(tmp_path: Path) -> None:
    repo_root, env = _build_cli_env(tmp_path)
    for run_type in ("training", "prediction", "evaluation", "rf_training"):
        config_path = tmp_path / f"{run_type}.json"
        if run_type == "training":
            write_json(config_path, {"dataset_name": "dataset-a"})
        elif run_type == "rf_training":
            write_json(config_path, {"dataset_name": "dataset-a"})
        elif run_type == "evaluation":
            write_json(config_path, {"dataset_name": "dataset-a", "sequence_length_ms": 400})
        else:
            model_path = tmp_path / "model.pt"
            model_path.write_bytes(b"")
            write_json(
                config_path,
                {
                    "model_path": str(model_path),
                    "mode": "tape",
                    "tape_files": [
                        {
                            "collection_name": "audio_collection_alpha",
                            "category_dir": "raw_recordings",
                            "relative_path": "tape_001.wav",
                        }
                    ],
                },
            )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alpaca_pipelines.cli",
                "create",
                run_type,
                "--config",
                str(config_path),
                "--json",
            ],
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert payload["run_type"] == run_type
        assert payload["status"] == "created"
        assert payload["run_id"]


def test_create_training_with_legacy_sequence_length_fails_validation(
    tmp_path: Path,
) -> None:
    repo_root, env = _build_cli_env(tmp_path)
    config_path = tmp_path / "training_legacy.json"
    write_json(config_path, {"dataset_name": "dataset-a", "sequence_length": 400})

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpaca_pipelines.cli",
            "create",
            "training",
            "--config",
            str(config_path),
            "--json",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "extra_forbidden" in result.stderr


def test_generate_slurm_json_does_not_mutate_lifecycle_state(tmp_path: Path) -> None:
    repo_root, env = _build_cli_env(tmp_path)
    config_path = tmp_path / "training.json"
    write_json(config_path, {"dataset_name": "dataset-a"})

    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpaca_pipelines.cli",
            "create",
            "training",
            "--config",
            str(config_path),
            "--json",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0, created.stderr
    created_payload = json.loads(created.stdout)
    run_id = str(created_payload["run_id"])

    generated = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpaca_pipelines.cli",
            "generate-slurm",
            "--run-id",
            run_id,
            "--json",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    generated_payload = json.loads(generated.stdout)
    assert generated_payload["run_id"] == run_id
    assert str(generated_payload["script_path"]).endswith("/slurm/job.sbatch")

    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpaca_pipelines.cli",
            "inspect",
            "--run-id",
            run_id,
            "--json",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert inspected.returncode == 0, inspected.stderr
    inspected_payload = json.loads(inspected.stdout)
    assert inspected_payload["status"] == "created"
    assert inspected_payload["submitted_at"] is None
    assert inspected_payload["slurm_job_id"] is None
