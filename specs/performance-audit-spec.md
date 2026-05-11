## Metadata-First Performance Plan for Run/Review Surfaces (alpaca-pipelines)

### Summary

- Add additive compact metadata artifacts for completed inference runs so downstream services can avoid request-path scans over detection payloads and file inventories.
- Keep existing persisted prediction/review contracts unchanged and authoritative.
- Provide explicit backfill tooling for legacy runs.

### Implemented in this repo

1. Additive compact review index artifact

- `prediction` and `rf_inference` executors now write:
  - `outputs/predictions/review_index_summary.json`
- Artifact includes:
  - run-level metadata (`run_id`, `run_type`, `rf_filtered`, `n_files`, `total_detections`)
  - per-file compact entries (`audio_file`, `n_windows`, `n_detections`)
  - RF partition totals when present (`accepted/rejected/unscored`) at run and per-file levels.

2. Run output pointer update

- `RunState.outputs` now includes `prediction_review_index_summary_path` for direct discovery by backend consumers.

3. Legacy-run backfill

- New CLI/API path:
  - `alpaca-pipelines backfill-review-index-summaries [--runs-root ...] [--run-id ...] [--json]`
- Backfill scans completed `prediction` and `rf_inference` runs and creates missing review index artifacts.

4. Contract + schema coverage

- Contract docs updated with `review_index_summary.json` shape and invariants.
- New JSON schema exported for `PredictionReviewIndexSummary`.
- Tests added for artifact build/backfill and CLI JSON contract.

### Compatibility and behavior

- Existing `prediction_summary.json`, per-file prediction JSON artifacts, and selection-table outputs remain unchanged.
- New artifact is additive and safe for incremental downstream adoption.
- Backfill is explicit; no hidden automatic migration on read paths.
