"""
Post-processing utilities for pipeline outputs.

Includes:
- Aggregation of evaluation results across multiple runs
- Export of detections to Raven selection table format
- Summary generation for reporting
"""

from __future__ import annotations

from datetime import datetime, timezone
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _selection_table_columns() -> list[str]:
    return [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Score",
    ]


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

    columns = _selection_table_columns()
    table = pd.DataFrame(selection_rows, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)

    return output_path


def export_prediction_run_selection_tables(
    predictions_dir: Path,
    selection_tables_dir: Path,
    freq_low_hz: int = 0,
    freq_high_hz: int = 4000,
    use_rf_filtered: bool = False,
) -> dict[str, Any]:
    """Export Raven selection tables for every audio file in a prediction run.

    Requires predictions_dir/prediction_summary.json to exist and to define
    the audio file list. For each audio file, exports a TSV under
    selection_tables_dir. Also writes a selection_tables_summary.json into
    selection_tables_dir.
    """
    if not predictions_dir.is_dir():
        raise FileNotFoundError("Predictions directory not found: {}".format(predictions_dir))

    summary_path = predictions_dir / "prediction_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Prediction summary not found: {}".format(summary_path))

    summary = read_json(summary_path)
    files = summary.get("files")
    if not isinstance(files, list):
        raise ValueError("prediction_summary.json missing 'files' list: {}".format(summary_path))

    audio_files: list[str] = []
    for entry in files:
        if not isinstance(entry, dict) or "audio_file" not in entry:
            raise ValueError(
                "Invalid prediction_summary.json entry "
                "(expected dict with 'audio_file'): {}".format(entry)
            )
        audio_file_value = entry["audio_file"]
        if not isinstance(audio_file_value, str) or not audio_file_value:
            raise ValueError(
                "Invalid audio_file value " "in prediction_summary.json: {}".format(entry)
            )
        audio_files.append(audio_file_value)

    stems: list[str] = [Path(audio_file).stem for audio_file in audio_files]
    if len(set(stems)) != len(stems):
        duplicates = sorted([s for s in set(stems) if stems.count(s) > 1])
        raise ValueError(
            "Prediction audio file stems are not unique (cannot export safely): {}".format(
                ", ".join(duplicates)
            )
        )

    selection_tables_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for audio_file, stem in zip(audio_files, stems, strict=True):
        suffix = "_rf_filtered" if use_rf_filtered else ""
        prediction_json_path = predictions_dir / "{}{}.json".format(stem, suffix)
        if not prediction_json_path.is_file():
            raise FileNotFoundError("Prediction output not found: {}".format(prediction_json_path))

        output_name = "{}{}.txt".format(stem, suffix)
        output_path = selection_tables_dir / output_name
        if output_path.exists():
            raise FileExistsError("Selection table output already exists: {}".format(output_path))

        export_detections_to_selection_table(
            predictions_path=prediction_json_path,
            output_path=output_path,
            freq_low_hz=freq_low_hz,
            freq_high_hz=freq_high_hz,
            use_rf_filtered=use_rf_filtered,
        )

        exported.append(
            {
                "audio_file": audio_file,
                "audio_file_stem": stem,
                "predictions_json": str(prediction_json_path),
                "selection_table": str(output_path),
            }
        )

    selection_tables_summary = {
        "generated_at": _now_iso(),
        "predictions_dir": str(predictions_dir),
        "selection_tables_dir": str(selection_tables_dir),
        "use_rf_filtered": use_rf_filtered,
        "freq_low_hz": freq_low_hz,
        "freq_high_hz": freq_high_hz,
        "n_files": len(exported),
        "files": exported,
    }

    summary_output_path = selection_tables_dir / "selection_tables_summary.json"
    write_json(summary_output_path, selection_tables_summary)

    return selection_tables_summary
