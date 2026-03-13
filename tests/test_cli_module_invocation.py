from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from alpaca_pipelines.io_utils import write_json


def test_python_module_invocation_executes_cli(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir()
    datasets_root.mkdir()
    runs_root.mkdir()
    write_json(collection_root / "merged_index.json", {"meta": {}, "entries": []})

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
