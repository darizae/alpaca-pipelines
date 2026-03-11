# SPEC: Day-Zero Filesystem Boundary for UI-Only Catalog DB

## Intent

Define the runtime contract when `alpaca-ui` uses PostgreSQL as a metadata catalog while `alpaca-pipelines` remains filesystem-driven.

## Ownership

`alpaca-pipelines` owns:

- run lifecycle and `run_state.json`
- workflow operation artifacts (`standardizer`, `dataset_builder`)
- output artifact generation and layout under `ALPACA_RUNS_ROOT`
- dataset artifacts under `ALPACA_DATASETS_ROOT`

`alpaca-ui` owns:

- catalog projection in PostgreSQL
- synchronization logic from HPC filesystem artifacts
- query APIs for projected metadata

## Hard boundary

`alpaca-pipelines` does not require, read, or write PostgreSQL.

Environment contract remains:

- `ALPACA_COLLECTION_ROOT`
- `ALPACA_MERGED_INDEX`
- `ALPACA_DATASETS_ROOT`
- `ALPACA_RUNS_ROOT`

## Day-zero rollout assumptions

- no backfill path is required
- no compatibility layer is required
- datasets/runs may be reset aggressively before first stable usage
- source WAV/FLAC and ZIP archives are preserved by reset runbooks outside this repo
