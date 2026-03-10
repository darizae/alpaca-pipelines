# SPEC: Run Result Contracts and Retention

## Goal / problem statement

`alpaca-ui` needs reliable, explicit access to run result artifacts for training, prediction, evaluation, RF training, and prediction postprocessing. At the same time, some run-generated artifacts are bulky and do not need to be retained forever.

The current architecture already makes `alpaca-pipelines` the owner of run-state and artifact layout. Retention policy must follow the same ownership boundary.

This spec defines:

- authoritative result artifacts per routine
- file and directory contracts relied on by `alpaca-ui`
- which artifacts are retained indefinitely
- which derived artifacts may be automatically pruned
- when pruning may occur
- how pruning outcomes are surfaced

## Ownership decision

`alpaca-pipelines` owns run-result contracts and derived-artifact retention.

`alpaca-ui` may display:

- artifact presence
- pruning history or status
- artifact-level warnings

`alpaca-ui` must not own:

- pruning policy
- pruning execution
- rules for deleting pipeline-owned run artifacts

Whole-run archival or deletion is out of scope for this spec.

## Shared environment inputs

This spec depends on the pipeline environment contract already documented in `../README.md` and `../contracts/CONTRACT.md`.

Relevant env vars:

- `ALPACA_COLLECTION_ROOT`
- `ALPACA_MERGED_INDEX`
- `ALPACA_DATASETS_ROOT`
- `ALPACA_RUNS_ROOT`

Current deployment default for `ALPACA_RUNS_ROOT` observed from `alpaca-ui/.env`:

- `/projects/extern/kisski/kisski-alpaca-2/dir.project/runs`

## Authoritative run artifact contracts

### Shared run files

For every run type, the following are authoritative:

- `ALPACA_RUNS_ROOT/<run_type>/<run_id>/run_state.json`
- `ALPACA_RUNS_ROOT/<run_type>/<run_id>/logs/`
- `ALPACA_RUNS_ROOT/<run_type>/<run_id>/outputs/`
- `ALPACA_RUNS_ROOT/<run_type>/<run_id>/slurm/`

### Training

Authoritative retained artifacts:

- `outputs/model/trained_model.pt`
- `outputs/summaries/` training summary artifacts
- `run_state.outputs.trained_model_path`
- `run_state.outputs.tensorboard_dir`

Intermediate artifacts that may be pruned:

- `outputs/model/checkpoints/`

Decision:

- training checkpoints are derived, potentially large, and not required once final model and retained summaries exist

### Prediction

Authoritative retained artifacts:

- `outputs/predictions/prediction_summary.json`
- per-file prediction JSON required by supported postprocessing workflows
- `run_state.outputs.predictions_dir`
- `run_state.outputs.prediction_selection_tables_summary_path` when exports exist

Conditionally retained artifacts:

- RF-filtered per-file prediction JSON when RF filtering was requested and is part of a supported export path

### Evaluation

Authoritative retained artifacts:

- `outputs/evaluation/evaluation_report.json`
- `run_state.outputs.evaluation_dir`

### RF training

Authoritative retained artifacts:

- `outputs/model/rf_model.joblib`
- `outputs/summaries/rf_training_report.json`
- `run_state.outputs.rf_model_path`

### Prediction selection-table export

Authoritative retained artifacts:

- `outputs/predictions/selection_tables/selection_tables_summary.json`
- exported `.txt` files listed in that summary

## Retention policy

### Retain indefinitely

Retain:

- `run_state.json`
- final model artifacts
- evaluation reports
- RF training reports
- prediction summary artifacts required for supported UI review and postprocessing
- selection-table export summary and exported files
- logs needed for diagnosis
- directory pointers in `RunState.outputs`

### Eligible for auto-pruning

Auto-pruning is allowed only for derived artifacts that are not authoritative outputs.

Initially allowed:

- training checkpoints under `outputs/model/checkpoints/`
- superseded selection-table export outputs only if a future export policy explicitly supports replacement and the summary file remains authoritative

Not allowed for auto-pruning under this spec:

- `run_state.json`
- final trained models
- RF models
- evaluation reports
- prediction summaries
- per-file prediction JSON needed by supported export workflows
- logs by default

### Out of scope

Out of scope for this spec:

- deleting whole run directories
- age-based archival of completed runs
- pruning user-owned external caches such as `cache_dir`
- deleting result artifacts merely because they are old

## Pruning execution rules

Hard rules:

- pruning must happen inside `alpaca-pipelines`
- pruning must run only after authoritative retained outputs for that routine exist
- pruning failure must not silently fall back to partial deletion behavior
- pruning must not infer safety heuristically when the contract does not authorize deletion

Decision:

- pruning should be deterministic and contract-driven, not best-effort cleanup by the UI backend

## Surfacing pruning state to the UI

`alpaca-ui` needs visibility, not ownership.

Required surfaced information:

- whether a run has prunable derived artifacts
- whether pruning has already occurred
- whether pruning failed
- which retained artifacts remain authoritative after pruning

Preferred contract approach:

- expose pruning state via a small explicit block in run summary or outputs metadata
- do not overload users with internal retention details when no action is needed

## UI dependencies on artifact contracts

The UI specs depend on these exact files:

- training reporting depends on summary artifacts under `outputs/summaries/`, final model pointer, and TensorBoard pointer
- prediction reporting depends on `outputs/predictions/prediction_summary.json`
- evaluation reporting depends on `outputs/evaluation/evaluation_report.json`
- RF training reporting depends on `outputs/summaries/rf_training_report.json`
- postprocessing export reporting depends on `outputs/predictions/selection_tables/selection_tables_summary.json`

These file locations are contract-level dependencies and must not change without updating both this spec and the UI-facing specs.

## Acceptance criteria

- ownership of retention remains entirely in `alpaca-pipelines`
- authoritative retained artifacts are explicit for every supported routine
- only derived artifacts are eligible for auto-pruning
- checkpoint pruning is explicitly allowed
- whole-run archival remains explicitly out of scope
- UI-facing dependencies on result files are documented exactly
- pruning outcomes can be surfaced to the UI without giving the UI deletion ownership

## Edge cases

- pruning requested before authoritative outputs exist: fail clearly
- expected retained artifact missing: do not prune related derived artifacts
- selection-table exports exist but replacement policy is not explicitly supported: do not delete old exports
- pruning encounters filesystem error: surface failure, keep run consistent, do not apply undocumented fallback behavior

## Tests and validation

`alpaca-pipelines` implementation branches derived from this spec must add:

- contract tests for retained vs prunable artifacts
- routine-specific artifact presence tests
- pruning eligibility tests
- pruning failure tests

Quality gates:

- `make lint`
- `make typecheck`
- `make test`
- `./.venv/bin/pre-commit run --all-files` when the environment is ready

Second-pass requirement:

- perform the second bug-gap review defined by the umbrella program spec in `alpaca-ui/specs/SPEC_PIPELINE_FEATURE_PROGRAM.md`
