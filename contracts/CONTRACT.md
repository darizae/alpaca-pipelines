# Contract: alpaca-pipelines

## Purpose
Mid-level orchestrator for bioacoustics deep learning pipelines.
Bridges the low-level `bioacoustics-dl-toolbox` mechanics with a
folder-based HPC persistence layer and a future backend/REST API.

## Scope
- Training CNN-based hum detectors
- Prediction (inference) on audio tapes or dataset test splits
- Evaluation of predictions against ground truth
- Optional RF filter post-processing on prediction results
- Post-processing and aggregation of pipeline outputs

## Hard rules

### No SQL; folder-based persistence only
All state is stored as JSON files under `ALPACA_RUNS_ROOT`.
Each run occupies a single directory containing its specification,
state, logs pointer, and outputs pointer.

### No dependency on alpaca-dataset-builder or alpaca-audio-standardizer
This repo reads `manifest.json`, `splits/*.csv`, and `merged_index.json`
as stable public interfaces.  It MUST NOT import or couple to those packages.

### No embedded paths
All paths come from explicit inputs (environment variables for Makefile,
explicit parameters for API).  No user-specific or cluster-specific
absolute paths are hardcoded.

### No fallback mechanisms
If a required input is missing or malformed, the system MUST fail
immediately with a clear error.  No silent salvage, no defaults for
required fields.

### Path safety
All relative paths resolved against any root MUST be validated to stay
inside that root.  Paths containing `..` or that are absolute MUST be
rejected.

### Run isolation
Each run is identified by a UUID and stored in its own directory.
Runs MUST NOT mutate each other's state.  Concurrent runs are safe
because they operate on separate directories.

### API-first design
All operations are available as programmatic methods on `PipelineAPI`.
The CLI is a thin wrapper around the API.  The future backend drives
the API directly.

### Immutable run specifications
Once a run is created, its specification MUST NOT change.  Status,
timestamps, and result pointers are the only mutable fields.

## Required environment variables
- `ALPACA_COLLECTION_ROOT`: root of audio collections
- `ALPACA_MERGED_INDEX`: path to merged_index.json
- `ALPACA_DATASETS_ROOT`: root of built datasets
- `ALPACA_RUNS_ROOT`: root for pipeline run state and outputs

## Run directory layout
```
ALPACA_RUNS_ROOT/
└── <run_type>/
    └── <run_id>/
        ├── run_state.json
        ├── logs/
        ├── outputs/
        │   ├── model/          (training)
        │   ├── predictions/    (prediction)
        │   ├── evaluation/     (evaluation)
        │   └── summaries/      (training, evaluation)
        └── slurm/
            └── job.sbatch
```

## Run state machine
```
created → submitted → running → completed
                             → failed
         → cancelled (from created or submitted)
```

## Persistence layer contracts (read-only inputs)

### merged_index.json
- `entries[].hum_path` is ROOT-RELATIVE to `ALPACA_COLLECTION_ROOT`
- Resolve as: `Path(ALPACA_COLLECTION_ROOT) / hum_path`

### manifest.json
- `snippets[].filename` is a basename under `<dataset_dir>/snippets/`
- `snippets[].source_path` is ROOT-RELATIVE to collection root
- `meta.collection_root` and `meta.merged_index_path` are informational

### splits CSV
- One filename per line (basename only)
- Resolve as: `<dataset_dir>/snippets/<filename>`
