# alpaca-pipelines

HPC-side workflow host for Alpaca bioacoustics processing.
Owns the contract, persistence, and execution boundary for collection standardization,
dataset building, training, prediction, evaluation, RF training, and postprocessing.

## Scope

- **Training** CNN-based hum detectors (ResNet encoder + classifier)
- **Prediction** (inference) on audio tapes via sliding window
- **Prediction manual review artifacts** (preview/batch spectrogram + clip generation, export copy)
- **Evaluation** of predictions against ground truth
- **RF filter** post-processing on prediction results
- **Collection standardization** (scan, rename planning/apply, index generation)
- **Raw AudioMoth import** into canonical collections with recorder sidecars
- **Dataset building** and review workflows
- **Post-processing** and export (Raven selection tables, aggregation)
- **SLURM** batch script generation for HPC execution
- **CLI JSON contract** for service-to-service integration
- **Run-state migration** from legacy `backend_meta.json` sidecars

## Architecture

```

┌─────────────────────────────────────────────────┐
│  alpaca-ui Backend / REST API                    │
│  (drives PipelineAPI programmatically)           │
├─────────────────────────────────────────────────┤
│  alpaca-pipelines                                │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ RunManager │  │ Executors │  │ SLURM Gen    │ │
│  │ (state)    │  │ (train/  │  │ (batch       │ │
│  │            │  │  predict/ │  │  scripts)    │ │
│  │            │  │  eval)    │  │              │ │
│  └───────────┘  └──────────┘  └──────────────┘ │
│  HPC persistence layer (folders + JSON)          │
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐ │
│  │ Collections │ │ Datasets   │ │ Runs root   │ │
│  │ (audio)     │ │ (snippets/ │ │ (state/     │ │
│  │             │ │  splits/)  │ │  outputs/)  │ │
│  └────────────┘ └────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘

````

## Installation

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install alpaca-pipelines
pip install -e ".[dev]"
pre-commit install
````

On the HPC, do not launch runs or workflow operations until the runtime preflight passes:

```bash
make runtime-check
```

That command fails if the venv exists but `alpaca_pipelines` is not actually installed from the current checkout.

## Configuration

Copy `.env.example` to `.env` and fill in the paths:

```bash
make env-init
# Edit .env with your paths
```

Required environment variables:

| Variable                 | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `ALPACA_COLLECTION_ROOT` | Root of audio collections (`audio_collection_*/`) |
| `ALPACA_MERGED_INDEX`    | Path to `merged_index.json`                       |
| `ALPACA_DATASETS_ROOT`   | Root of built datasets (strategy directories)     |
| `ALPACA_RUNS_ROOT`       | Root for pipeline run state and outputs           |

UI-only catalog boundary:

- `alpaca-pipelines` remains filesystem-only and database-unaware.
- If `alpaca-ui` uses PostgreSQL for metadata projection, it must ingest from these filesystem contracts.
- Run deletion is backend-owned in `alpaca-ui`: delete the run directory under `ALPACA_RUNS_ROOT/<run_type>/<run_id>` and prune UI metadata in Postgres. `alpaca-pipelines` does not expose a run-delete command.

## HPC runtime preflight

Before using `alpaca-ui` against a remote HPC checkout, verify the remote runtime from inside the `alpaca-pipelines` repo:

```bash
make env-check
make runtime-check
```

What `make runtime-check` verifies:

- `.venv/bin/python` exists
- `.venv/bin/alpaca-pipelines` exists
- `alpaca_pipelines` imports from the current checkout
- the CLI shebang points into the same checkout
- the required `ALPACA_*` variables allow CLI startup
- `dataset-status --json` runs successfully

If it fails, rebuild the runtime in place and re-run the preflight:

```bash
rm -rf .venv
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev]"
make runtime-check
```

On the HPC, `/projects/extern/...` may resolve to a canonical `/mnt/vast-kisski/...` path. That is acceptable as long as the venv and imported package still resolve to the same checkout.

Do not launch `dataset-build`, `dataset-prepare-review`, `dataset-apply-review`, `standardizer-import`, `standardizer-scan`, `standardizer-plan`, `standardizer-apply`, or `standardizer-index` until this preflight succeeds.

## Usage

### CLI

```bash
# Create a training run
make create-training-run RUN_CONFIG=configs/training_example.json

# Submit a created run
alpaca-pipelines submit --run-id <uuid>

# Execute a run
make execute-run RUN_ID=<uuid>

# List all runs
make list-runs

# Inspect a run
make inspect-run RUN_ID=<uuid>

# Generate SLURM script
make generate-slurm RUN_ID=<uuid>

# Migrate legacy submission metadata into run_state.json
alpaca-pipelines migrate-backend-meta --json

# Generate a one-item manual-review preview from a session manifest
alpaca-pipelines prediction-review-preview --manifest /path/to/session_manifest.json --item-id <item-id> --json

# Generate all manual-review artifacts for a session manifest
alpaca-pipelines prediction-review-generate --manifest /path/to/session_manifest.json --json

# Export generated manual-review artifacts (single item or full session)
alpaca-pipelines prediction-review-export --manifest /path/to/session_manifest.json --destination-dir /path/to/export --item-id <item-id> --json
```

### CLI JSON mode

The UI-facing commands support `--json` and write exactly one JSON document to stdout:

- `create`
- `submit`
- `inspect`
- `list`
- `cancel`
- `generate-slurm`
- `export-selection-tables`
- `migrate-backend-meta`
- `prediction-review-preview`
- `prediction-review-generate`
- `prediction-review-export`
- `standardizer-scan`
- `standardizer-import`
- `standardizer-plan`
- `standardizer-apply`
- `standardizer-index`
- `standardizer-job`
- `standardizer-status`
- `dataset-build`
- `dataset-prepare-review`
- `dataset-apply-review`
- `dataset-job`
- `dataset-status`
- `fail-operation`
- `delete-failed-operation`

For run lifecycle integration, `alpaca-ui` uses `create --json`, `submit --json`, and `cancel --json`. Run deletion is handled by `alpaca-ui` outside the `alpaca-pipelines` CLI.

### Prediction manual-review artifact contract

`alpaca-pipelines` provides a dedicated manual-review spectrogram schema
(`PredictionReviewSpectrogramConfig`) that is separate from training/evaluation
spectrogram settings.

Default values (also in [`configs/prediction_review_spectrogram_example.json`](configs/prediction_review_spectrogram_example.json)):

- Hann window (`window_function="hann"`)
- `window_size_samples=2002`
- `hop_size_samples=1001` (50% overlap)
- `dft_size=2048`
- `clipping_enabled=false`
- `averaging=1`
- `auto_apply=false`
- `colormap="magma"`
- labeled axes (`Time (s)`, `Frequency (kHz)`) and colorbar enabled

Artifact layout is deterministic per prediction run/session:

`ALPACA_RUNS_ROOT/prediction/<run_id>/outputs/manual_review/<session_id>/...`

### Standardizer workflow

The standardizer is now phase-based:

1. `standardizer-import` imports raw batch directories like `401_m28_20250213` into canonical `audio_collection_<batch>` collections.
2. `standardizer-scan` reports explicit capability/status for every collection:
   `raw_only`, `clips_only`, `hums_only`, `ready`, or `empty`.
3. `standardizer-plan` includes every collection and now plans canonical renames for:
   - labelled clips
   - segmented hums
   - raw `raw_recordings/*.WAV` and `raw_recordings/*.CSV`
4. Raw canonicalization uses `<subject_id>_<YYYYMMDD>_<HHMMSS>.(WAV|CSV)` and enforces strict `recordings.json` consistency:
   - plan fails if any raw WAV/CSV file is not represented in `recordings.json`
5. `standardizer-apply` applies filesystem renames and `recordings.json` path updates as one operation with rollback coverage.
6. `standardizer-index` emits a per-collection index artifact for every collection.
   Raw-only collections get `entries=[]` plus `recordings` metadata, and merged recordings include all collections.

Raw-only collections remain valid inference sources. Prediction supports a
collection mode that resolves `.wav` files from selected `audio_collection_*`
directories (for example `raw_recordings/`) without requiring labeled clips.

Dataset build still derives positive targets from hum index entries only. Raw
audio remains a valid negative/noise source. If no labelled targets are
available, `dataset-build` fails explicitly.

Imported raw collections store source files under:

```text
audio_collection_<batch>/
├── raw_recordings/
│   ├── <stem>.WAV
│   ├── <stem>.CSV              # optional
│   ├── DEVICE.TXT              # optional
│   ├── SETTINGS.txt            # optional
│   └── LOG.TXT                 # optional
└── recordings.json
```

`recordings.json`, `merged_index.json`, and dataset `manifest.json` may now include additive recorder metadata:

- source-recording summaries
- optional per-second AudioMoth track points
- `source_recording_key` joins from hums/snippets back to their source WAV
- snippet-derived absolute timestamps and GPS midpoint fields when sidecars exist

### Prediction input modes

Prediction now supports three explicit input modes:

1. `tape`: explicit `tape_files` handle list (`collection_name`, `category_dir`, `relative_path`).
2. `dataset`: dataset test split via `dataset_name`.
3. `collection`: `collection_names` + `source_category_dirs` (for unlabeled collections).

### Post-processing: Raven selection tables (prediction runs)

After a **prediction** run completes, you can export **Raven-compatible** selection
tables (TSV `.txt`) for **all predicted audio files** in that run. This is a batch
operation driven by the run’s `prediction_summary.json` (authoritative list of files).

Outputs are persisted under:

`ALPACA_RUNS_ROOT/prediction/<run_id>/outputs/predictions/selection_tables/`

#### Makefile

```bash
# Export selection tables from base prediction outputs (<stem>.json)
make export-prediction-selection-tables RUN_ID=<prediction-run-uuid>

# Export selection tables from RF-filtered outputs (<stem>_rf_filtered.json)
make export-prediction-selection-tables RUN_ID=<prediction-run-uuid> USE_RF_FILTERED=1

# Override frequency bounds in the exported tables
make export-prediction-selection-tables RUN_ID=<prediction-run-uuid> FREQ_LOW_HZ=0 FREQ_HIGH_HZ=4000
```

#### CLI (direct)

```bash
# Base mode
alpaca-pipelines export-selection-tables --run-id <prediction-run-uuid>

# RF-filtered mode
alpaca-pipelines export-selection-tables --run-id <prediction-run-uuid> --use-rf-filtered

# With explicit frequency bounds
alpaca-pipelines export-selection-tables --run-id <prediction-run-uuid> --freq-low-hz 0 --freq-high-hz 4000
```

Hard behavior:

* The prediction run must be `completed`.
* The exporter uses `prediction_summary.json` and will not infer files by directory scanning.
* Missing required prediction JSON files cause immediate failure.
* If output `.txt` files already exist, the exporter fails immediately.

### Python API

```python
from alpaca_pipelines import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.training.config import TrainingRunSpec

# Initialize
env = PipelineEnvironment.from_explicit(
    collection_root="/path/to/collections",
    merged_index_path="/path/to/merged_index.json",
    datasets_root="/path/to/datasets",
    runs_root="/path/to/runs",
)
api = PipelineAPI(env)

# Create and execute a training run
spec = TrainingRunSpec(dataset_name="clipwise_balanced_nph_2")
run = api.create_training_run(spec)
print(f"Created: {run.run_id}")

# Poll status
status = api.get_run_status(run.run_id)
print(f"Status: {status.status}")

# Execute
result = api.execute_run(run.run_id)
print(f"Completed: {result.outputs.trained_model_path}")
```

### Clearing a stale workflow job

If a dataset-builder or standardizer operation is stuck in `pending` or `running` with no real worker behind it, mark it failed explicitly:

```bash
make clean-stale-workflow-job JOB_ID=<workflow-job-id>
```

This writes a terminal `failed` state into the operation record so status surfaces stop reporting it as active.

### Deleting a failed workflow job

Once a workflow job is already in terminal `failed` state, remove its operation directory explicitly:

```bash
make delete-failed-workflow-job JOB_ID=<workflow-job-id>
```

This deletes the failed job directory under `ALPACA_RUNS_ROOT/operations/...`. It refuses to delete jobs unless their current status is `failed`.

#### Python API: selection tables export (batch)

```python
from alpaca_pipelines import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment

env = PipelineEnvironment.from_explicit(
    collection_root="/path/to/collections",
    merged_index_path="/path/to/merged_index.json",
    datasets_root="/path/to/datasets",
    runs_root="/path/to/runs",
)
api = PipelineAPI(env)

summary = api.export_prediction_run_selection_tables(
    prediction_run_id="<prediction-run-uuid>",
    freq_low_hz=0,
    freq_high_hz=4000,
    use_rf_filtered=False,
)

print(summary["selection_tables_dir"])
```

## Run State Machine

```
created → submitted → running → completed
                             → failed
         → cancelled (from created or submitted)
```

Run deletion is external to this state machine. When an external orchestrator deletes a run directory, that run simply disappears from list/inspect surfaces because `run_state.json` no longer exists.

Each run is stored as a folder under `ALPACA_RUNS_ROOT/<run_type>/<run_id>/`
containing `run_state.json`, `logs/`, `outputs/`, and `slurm/`.

`submitted_at` and `slurm_job_id` are persisted by `alpaca-pipelines` in `run_state.json`.
`backend_meta.json` is a legacy migration input only and is not part of the supported run-state contract.

### Output contract by run type

Output paths in `run_state.json` are routine-specific and should only be populated for the active run type.

- `training`
  - `outputs/model/`
  - `outputs/summaries/`
  - `outputs/summaries/training_summary.json`
  - `outputs/summaries/training_history.json`
  - `outputs.tensorboard_dir`
- `prediction`
  - `outputs/predictions/`
  - `outputs/predictions/prediction_summary.json`
  - `outputs/predictions/selection_tables/selection_tables_summary.json` after export
- `evaluation`
  - `outputs/evaluation/`
  - `outputs/evaluation/evaluation_report.json`
- `rf_training`
  - `outputs/model/`
  - `outputs/summaries/`
  - `outputs/summaries/rf_training_report.json`

For training runs, `run_state.progress` must reflect terminal progress on completion, including:

- `current_epoch`
- `total_epochs`
- `current_phase`
- `best_metric_name`
- `best_metric_value`

The UI may recover older completed training runs from structured logs, but that is compatibility for existing historical runs, not the output contract for new runs.

## Contracts

Committed JSON Schemas for the UI-facing contracts live under `contracts/json-schema/`:

- `TrainingRunSpec.json`
- `PredictionRunSpec.json`
- `EvaluationRunSpec.json`
- `RfTrainingRunSpec.json`
- `SlurmConfig.json`
- `RunState.json`

## Development

```bash
make lint       # Run ruff
make format     # Auto-format
make fix        # Ruff format + ruff check --fix
make typecheck  # Run mypy
make test       # Run pytest
make check      # fix, then mypy and pytest
```
