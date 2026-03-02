"""
Post-processing utilities for pipeline outputs.

Includes:
- Aggregation of evaluation results across multiple runs
- Export of detections to Raven selection table format
- Summary generation for reporting
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from alpaca_pipelines.io_utils import read_json, write_json


def aggregate_evaluation_results(
    evaluation_dirs: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Aggregate evaluation reports from multiple runs into a comparison table.

    Reads evaluation_report.json from each directory and produces
    a combined summary.
    """
    rows: list[dict[str, Any]] = []
    for evaluation_dir in evaluation_dirs:
        report_path = evaluation_dir / "evaluation_report.json"
        if not report_path.is_file():
            raise FileNotFoundError("Evaluation report not found: {}".format(report_path))

        report = read_json(report_path)
        results = report["results"]
        metrics = results["metrics"]

        rows.append(
            {
                "run_id": report["run_id"],
                "dataset": report["dataset_name"],
                "split": results["split"],
                "n_samples": results["n_samples"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "auc": metrics["auc"],
            }
        )

    comparison = {
        "n_runs": len(rows),
        "runs": rows,
    }

    write_json(output_path, comparison)
    return comparison


def export_detections_to_selection_table(
    predictions_path: Path,
    output_path: Path,
    freq_low_hz: int = 0,
    freq_high_hz: int = 4000,
    use_rf_filtered: bool = False,
) -> Path:
    """Export detections from a prediction JSON file to a Raven selection table.

    The output is a tab-separated file compatible with Raven Pro,
    with columns: Selection, View, Channel, Begin Time (s), End Time (s),
    Low Freq (Hz), High Freq (Hz), Score.
    """
    if not predictions_path.is_file():
        raise FileNotFoundError("Predictions file not found: {}".format(predictions_path))

    prediction_data = read_json(predictions_path)
    detections = prediction_data.get("detections", [])

    selection_rows: list[dict[str, object]] = []
    selection_counter = 1
    for detection in detections:
        if use_rf_filtered and not detection.get("rf_pass", True):
            continue

        score = detection.get("rf_score") if use_rf_filtered else detection.get("score")

        selection_rows.append(
            {
                "Selection": selection_counter,
                "View": "Spectrogram 1",
                "Channel": 1,
                "Begin Time (s)": detection["start_s"],
                "End Time (s)": detection["end_s"],
                "Low Freq (Hz)": freq_low_hz,
                "High Freq (Hz)": freq_high_hz,
                "Score": score,
            }
        )
        selection_counter += 1

    table = pd.DataFrame(selection_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)

    return output_path
