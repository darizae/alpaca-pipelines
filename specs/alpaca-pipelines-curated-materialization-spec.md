# alpaca-pipelines Spec: Curated Prediction Materialization Contract Alignment

## Purpose

Ensure `alpaca-pipelines` can consume curated examples exported by `alpaca-ui` and materialize them into the existing durable curated source root used by Dataset Builder.

The extraction/materialization flow already exists in `alpaca-pipelines` via:

- `prediction-review-materialize-curated`
- `curated-source-status`
- `PipelineAPI.materialize_curated_prediction_examples(...)`
- `PipelineAPI.list_curated_prediction_sources(...)`

Dataset Builder already supports curated examples through:

- `include_manual_review_curated`
- `manual_review_curated_filters`
- `manual_review_curated_max_examples`

Therefore, no new materialization pipeline or dataset-building path is required.

## Current behavior

`prediction-review-materialize-curated` accepts either:

1. `--manifest` + `--labels`, or
2. `--curated-export-manifest`

The materializer writes snippets and manifests under:

```text
{ALPACA_DATASETS_ROOT}/_curated_prediction_examples/
```

Dataset Builder reads that root when `include_manual_review_curated = true`.

## Required change

Align the curated export manifest schema with `alpaca-ui`.

### 1. Accept `curated_example_id`

Update `CuratedPredictionExportItem` in:

```text
src/alpaca_pipelines/prediction/review/curated.py
```

Add:

```python
curated_example_id: str | None = None
```

When materializing:

```python
if item.curated_example_id:
    curated_example_id = item.curated_example_id
else:
    curated_example_id = _build_curated_example_id(
        prediction_run_id=prediction_run_id,
        review_session_id=review_session_id,
        review_item_id=item.review_item_id,
    )
```

Reason: `alpaca-ui` already has durable DB IDs for curated examples. The materialized source manifest should preserve those IDs exactly so UI DB rows, materialized snippets, dataset manifests, and later provenance all refer to the same curated example.

### 2. Keep canonical field name `label`

`CuratedPredictionExportItem` should continue to require:

```python
label: Literal["target", "noise"]
```

Do not add a `curation_label` alias unless needed for backwards compatibility. `alpaca-ui` should emit `label`.

### 3. Preserve existing materialization layout

Do not change the destination layout:

```text
{datasets_root}/_curated_prediction_examples/
  {collection_name}/
    {prediction_run_id}/
      {review_session_id}/
        {source_recording_key}/
          manifest.json
          snippets/
            target_<curated_example_id>.wav
            noise_<curated_example_id>.wav
```

The existing layout is already consumed by Dataset Builder.

### 4. No new CLI command

Do not add a new command such as `curated-export-to-collection`.

Continue using:

```bash
python -m alpaca_pipelines.cli prediction-review-materialize-curated \
  --curated-export-manifest <manifest.json> \
  --json
```

### 5. Expected `alpaca-ui` export manifest

The manifest consumed by `--curated-export-manifest` should support this shape:

```json
{
  "schema_version": 1,
  "prediction_run_id": "prediction-run-id",
  "review_session_id": "review-session-id",
  "source_collection_name": "audio_collection_x",
  "source_category_dir": "raw_recordings",
  "source_relative_path": "path/to/source.wav",
  "source_audio_file": "/absolute/or/pipeline-known/source.wav",
  "items": [
    {
      "curated_example_id": "ui-curated-example-id",
      "review_item_id": "123",
      "source_audio_file": "/absolute/or/pipeline-known/source.wav",
      "start_s": 12.34,
      "end_s": 13.56,
      "label": "target",
      "detection_index": 42,
      "detection_score": 0.91,
      "source_collection_name": "audio_collection_x",
      "source_category_dir": "raw_recordings",
      "source_relative_path": "path/to/source.wav",
      "payload_json": {
        "alpaca_ui_curated_example_id": "ui-curated-example-id"
      }
    }
  ]
}
```

### 6. Tests

Add or update tests for:

- `CuratedPredictionExportItem` accepts `curated_example_id`.
- Materialization uses supplied `curated_example_id` when present.
- Materialization falls back to hash-based ID when absent.
- `prediction-review-materialize-curated --curated-export-manifest ... --json` still succeeds.
- Dataset Builder still selects materialized curated examples when `include_manual_review_curated = true`.
- Existing review-manifest + labels mode remains unchanged.

## Acceptance criteria

- A UI-generated curated export manifest validates successfully.
- Materialized manifests contain the exact `curated_example_id` supplied by `alpaca-ui`.
- Dataset Builder includes materialized curated examples when configured.
- Existing materialization modes and dataset-building tests continue to pass.
