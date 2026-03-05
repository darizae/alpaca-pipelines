# alpaca-pipelines

Mid-level orchestrator for bioacoustics deep learning pipelines.
Bridges the low-level `bioacoustics-dl-toolbox` mechanics with a
folder-based HPC persistence layer and a future backend/REST API.

## Scope

- **Training** CNN-based hum detectors (ResNet encoder + classifier)
- **Prediction** (inference) on audio tapes via sliding window
- **Evaluation** of predictions against ground truth
- **RF filter** post-processing on prediction results
- **Post-processing** and export (Raven selection tables, aggregation)
- **SLURM** batch script generation for HPC execution

## Architecture

```

┌─────────────────────────────────────────────────┐
│  Future Backend / REST API                       │
│  (drives PipelineAPI programmatically)           │
├─────────────────────────────────────────────────┤
│  alpaca-pipelines                                │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ RunManager │  │ Executors │  │ SLURM Gen    │ │
│  │ (state)    │  │ (train/  │  │ (batch       │ │
│  │            │  │  predict/ │  │  scripts)    │ │
│  │            │  │  eval)    │  │              │ │
│  └───────────┘  └──────────┘  └──────────────┘ │
├─────────────────────────────────────────────────┤
│  bioacoustics-dl-toolbox (low-level mechanics)   │
├─────────────────────────────────────────────────┤
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

# Install bioacoustics-dl-toolbox (local dev)
pip install -e /path/to/bioacoustics-dl-toolbox

# Install alpaca-pipelines
pip install -e ".[dev]"
pre-commit install
````

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

## Usage

### CLI

```bash
# Create a training run
make create-training-run RUN_CONFIG=configs/training_example.json

# Execute a run
make execute-run RUN_ID=<uuid>

# List all runs
make list-runs

# Inspect a run
make inspect-run RUN_ID=<uuid>

# Generate SLURM script
make generate-slurm RUN_ID=<uuid>
```

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

Each run is stored as a folder under `ALPACA_RUNS_ROOT/<run_type>/<run_id>/`
containing `run_state.json`, `logs/`, `outputs/`, and `slurm/`.

## Development

```bash
make lint       # Run ruff
make format     # Auto-format
make typecheck  # Run mypy
make test       # Run pytest
```
