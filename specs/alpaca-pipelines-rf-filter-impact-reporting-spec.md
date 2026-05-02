# Spec: RF Filter Impact Reporting

Repository: `darizae/alpaca-pipelines`

## Objective

Add authoritative RF filter impact reporting to prediction runs.

Today RF filtering writes `*_rf_filtered.json` files containing per-detection `rf_score` and `rf_pass`, and the executor logs per-file pass counts. The prediction summary does not persist aggregate RF impact counts, so downstream UI cannot reliably show how many detections were kept or rejected.

After this change, every RF-filtered prediction run must persist an `rf_filter_summary` inside `outputs/predictions/prediction_summary.json`.

No backward compatibility is required.

## Current behavior audit

- Base prediction JSON files are written first.
- RF filtering writes sibling `*_rf_filtered.json` files.
- Base files are not overwritten.
- RF-filtered detections retain all detections and annotate each with `rf_score` and `rf_pass`.
- Selection-table export can export base or RF-filtered outputs.
- When RF-filtered export is requested, detections with `rf_pass=false` are skipped.

## Desired UX-enabling data

Persist this object in `prediction_summary.json` when RF filtering is applied:

```json
{
  "rf_filter_summary": {
    "applied": true,
    "rf_model_path": "/path/to/rf_model.joblib",
    "rf_threshold": 0.4,
    "base_detections": 1240,
    "rf_passed": 312,
    "rf_rejected": 928,
    "rf_unscored": 0,
    "rejection_rate": 0.748387,
    "pass_rate": 0.251613,
    "files": [
      {
        "audio_file": "/path/audio.wav",
        "prediction_file": "/path/audio_hash.json",
        "rf_filtered_file": "/path/audio_hash_rf_filtered.json",
        "base_detections": 31,
        "rf_passed": 8,
        "rf_rejected": 23,
        "rf_unscored": 0,
        "rejection_rate": 0.741935,
        "pass_rate": 0.258065
      }
    ]
  }
}
```

When RF filtering is not applied, omit `rf_filter_summary`.

## Files to change

### 1. RF filter executor

Path:

```text
src/alpaca_pipelines/rf/executor.py
```

Change `apply_rf_filter(...)` from returning `None` to returning `dict[str, Any]`.

New signature:

```python
def apply_rf_filter(
    prediction_inputs: list[dict[str, str]],
    rf_model_path: str,
    rf_threshold: float,
    rf_feature_config: dict[str, Any] | None,
    prediction_logger: logging.Logger,
) -> dict[str, Any]:
    ...
```

While processing each prediction file, compute:

```text
base_detections = len(detections)
rf_passed = number of detections where rf_pass is True
rf_rejected = number of detections where rf_pass is False and rf_score is not None
rf_unscored = number of detections where rf_score is None
```

Per-file summary entry:

```python
{
    "audio_file": audio_file,
    "prediction_file": str(prediction_file),
    "rf_filtered_file": str(filtered_path),
    "base_detections": base_detections,
    "rf_passed": rf_passed,
    "rf_rejected": rf_rejected,
    "rf_unscored": rf_unscored,
    "rejection_rate": round(rf_rejected / base_detections, 6) if base_detections else 0.0,
    "pass_rate": round(rf_passed / base_detections, 6) if base_detections else 0.0,
}
```

Aggregate summary:

```python
{
    "applied": True,
    "rf_model_path": rf_model_path,
    "rf_threshold": rf_threshold,
    "base_detections": total_base,
    "rf_passed": total_passed,
    "rf_rejected": total_rejected,
    "rf_unscored": total_unscored,
    "rejection_rate": round(total_rejected / total_base, 6) if total_base else 0.0,
    "pass_rate": round(total_passed / total_base, 6) if total_base else 0.0,
    "files": file_summaries,
}
```

Keep current side effects:

```text
- Write *_rf_filtered.json.
- Keep all detections in filtered JSON.
- Keep per-detection rf_score and rf_pass.
- Keep file-level rf_filtered, rf_model_path, rf_threshold.
- Keep logging "RF filter <file>: passed/total detections passed".
```

### 2. Prediction executor

Path:

```text
src/alpaca_pipelines/prediction/executor.py
```

Currently prediction writes base files, optionally applies RF filter, then writes `prediction_summary.json`.

Change the RF section:

```python
rf_filter_summary: dict[str, Any] | None = None

if spec.apply_rf_filter and spec.rf_model_path is not None:
    ...
    rf_filter_summary = apply_rf_filter(...)
    run_manager.update_outputs(run_state.run_id, rf_filtered=True)
```

When building `summary`, include:

```python
"rf_filtered": bool(rf_filter_summary),
"rf_filter_summary": rf_filter_summary,
```

Keep existing fields:

```text
run_id
model_path
n_files
total_detections
detection_threshold
files
```

Important semantics:

```text
total_detections remains the base CNN detection count.
rf_filter_summary.rf_passed is the RF-filtered kept count.
rf_filter_summary.rf_rejected is the removed/rejected count.
```

### 3. Postprocessing export summary

Path:

```text
src/alpaca_pipelines/postprocessing/executor.py
```

In `export_prediction_run_selection_tables(...)`, enhance `selection_tables_summary.json`.

Add:

```json
{
  "source_mode": "base",
  "n_exported_detections": 312
}
```

Use `"source_mode": "rf_filtered"` when `use_rf_filtered=true`.

Implementation:

- Make `export_detections_to_selection_table(...)` return row count with the output path, or add a helper to count rows before writing.
- Sum exported rows across files.
- Keep existing summary fields.

Optional per-file export count:

```json
{
  "audio_file": "...",
  "audio_file_stem": "...",
  "predictions_json": "...",
  "selection_table": "...",
  "n_exported_detections": 8
}
```

### 4. Tests

Add or update tests under:

```text
tests/
```

Required tests:

```text
test_apply_rf_filter_returns_impact_summary
test_prediction_summary_includes_rf_filter_summary
test_rf_filter_summary_counts_pass_reject_unscored
test_selection_table_export_reports_base_exported_detection_count
test_selection_table_export_reports_rf_filtered_exported_detection_count
test_base_and_rf_filtered_outputs_both_remain_available
```

Test expectations:

```text
- Base prediction JSON still exists.
- RF-filtered JSON exists separately.
- prediction_summary.json has total_detections equal to base detections.
- prediction_summary.json has rf_filter_summary.rf_passed and rf_rejected.
- RF-filtered export skips rf_pass=false detections.
- Base export includes all base detections.
```

### 5. Acceptance criteria

A completed RF-filtered prediction run must produce:

```text
outputs/predictions/prediction_summary.json
```

with:

```text
total_detections = base CNN detections
rf_filtered = true
rf_filter_summary.base_detections = base CNN detections
rf_filter_summary.rf_passed = kept detections
rf_filter_summary.rf_rejected = rejected detections
rf_filter_summary.rf_unscored = unscored detections
rf_filter_summary.rejection_rate
rf_filter_summary.pass_rate
rf_filter_summary.files[]
```

Selection-table export must continue supporting both:

```text
Base export
RF-filtered export
```

and must report how many detections were exported.

## Out of scope

- Do not change RF scoring logic.
- Do not delete or overwrite base prediction outputs.
- Do not change the meaning of `detection_threshold` or `rf_threshold`.
- Do not add UI code in this repository.
