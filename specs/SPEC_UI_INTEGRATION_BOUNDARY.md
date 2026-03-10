# SPEC: alpaca-ui Integration Boundary for alpaca-pipelines

## Goal / problem statement

`alpaca-ui` currently treats `alpaca-pipelines` in two different ways at once:

- as a remote CLI that is invoked on the HPC over SSH
- as a Python package whose internal contracts are mirrored or depended on locally

That split is weaker than the integration used for `alpaca-audio-standardizer` and `alpaca-dataset-builder`.

The main architectural problem is not that `alpaca-pipelines` owns its own run state. That part is correct. The real problem is that `alpaca-ui` currently bypasses part of that ownership:

- `alpaca-pipelines` defines `submitted` state, `submitted_at`, and `slurm_job_id` in its own run contract
- `alpaca-ui` does not use `alpaca-pipelines` to record submission
- `alpaca-ui` writes separate `backend_meta.json` metadata next to `run_state.json`
- `alpaca-ui` parses human CLI output to recover the created `run_id`

This spec proposes a contract-first integration boundary so that `alpaca-ui` no longer needs `alpaca-pipelines` as a Python dependency, while `alpaca-pipelines` remains the sole owner of pipeline run lifecycle state.

This spec is intended to be complete enough to implement directly. It leaves no architectural decision open for later interpretation.

## Validated current HPC state

Validated by read-only inspection over SSH via `gwdg-kisski` on March 10, 2026.

Observed deployment facts:

- login host resolved successfully
- Slurm client commands are available on the login environment:
  - `/usr/local/slurm/current/install/bin/sbatch`
  - `/usr/local/slurm/current/install/bin/scancel`
  - `/usr/local/slurm/current/install/bin/squeue`
- `/projects/extern/kisski/kisski-alpaca-2/dir.project` exists and is a symlink to `/mnt/vast-kisski/projects/kisski-alpaca-2`
- these directories exist and are populated:
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/config`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/data`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/datasets`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/runs`
- these repositories and venv entrypoints exist on the HPC:
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-pipelines`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-pipelines/.venv/bin/alpaca-pipelines`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-audio-standardizer`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-audio-standardizer/.venv/bin/alpaca-audio`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-dataset-builder`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-dataset-builder/.venv/bin/alpaca-dataset`
- a remote `alpaca-ui` checkout was **not** present at `/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-ui`
- the actual checked `.env` files on the HPC are:
  - `alpaca-pipelines/.env`
    - `ALPACA_COLLECTION_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection`
    - `ALPACA_MERGED_INDEX=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json`
    - `ALPACA_DATASETS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/datasets`
    - `ALPACA_RUNS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/runs`
  - `alpaca-dataset-builder/.env`
    - `ALPACA_COLLECTION_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection`
    - `ALPACA_MERGED_INDEX=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json`
    - `ALPACA_DATASETS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/datasets`
  - `alpaca-audio-standardizer/.env`
    - `ALPACA_AUDIO_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/_work_copies/raw_audio_collection_work_20260301_172055`
    - `ALPACA_AUDIO_WORK_PARENT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/_work_copies`
    - `ALPACA_IDENTITY_MAP=contracts/IDENTITY_MAP.example.json`
- the persistence inputs used by the current UI config exist:
  - `ALPACA_COLLECTION_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection`
  - `ALPACA_MERGED_INDEX=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json`
  - `ALPACA_IDENTITY_MAP_PATH=/projects/extern/kisski/kisski-alpaca-2/dir.project/config/identity_map.json`
- `ALPACA_RUNS_ROOT` currently contains:
  - pipeline-owned runs under `training/` and `prediction/`
  - UI-owned durable jobs under `alpaca-ui-jobs/`
- at least one existing training run contains both:
  - `run_state.json` with `submitted_at=null` and `slurm_job_id=null`
  - `backend_meta.json` with the actual `submitted_at` and `slurm_job_id`

The last point is decisive: migration must include a one-time backfill of existing pipeline runs that still depend on sidecar submission metadata.

The checked `.env` contents also show that the three HPC-side repos do not share one uniform local `.env` contract. In particular, `alpaca-audio-standardizer/.env` is configured for a work-copy workflow and is not the source of truth for the UI integration.

## Critical assessment

It does **not** make sense to force `alpaca-pipelines` into the same implementation shape as `alpaca-audio-standardizer` and `alpaca-dataset-builder`.

Those tools are different in one important way:

- `alpaca-audio-standardizer` and `alpaca-dataset-builder` expose mostly single-shot filesystem workflows
- they do not own a durable async job model for the UI
- `alpaca-ui` therefore adds that job model itself via `remote_job_service`

`alpaca-pipelines` is not in that category:

- it already owns durable per-run directories under `ALPACA_RUNS_ROOT`
- it already owns `run_state.json`
- it already defines a state machine with `created`, `submitted`, `running`, `completed`, `failed`, `cancelled`
- it already owns execution semantics and Slurm script generation

Because of that, replacing pipeline run ownership with `alpaca-ui`'s `remote_job_service` would be a regression:

- duplicated state machines
- duplicated persistence
- split ownership over the same long-running operation
- harder reasoning about source of truth

The right alignment is therefore at the **integration boundary**, not at the **internal implementation** level.

## Decision

`alpaca-pipelines` should remain the owner of:

- run creation
- run persistence
- submission metadata
- cancellation semantics
- `run_state.json`

`alpaca-ui` should interact with it the same way it interacts with the other HPC-side tools:

- through a stable remote CLI boundary
- through documented on-disk contracts
- without importing `alpaca-pipelines` Python modules

This means:

- `alpaca-ui` remains responsible for SSH transport, remote temp file handling, and HTTP presentation
- `alpaca-pipelines` remains responsible for validating, mutating, and persisting pipeline run lifecycle state

## Ownership rules

### `alpaca-pipelines` owns

- validation of pipeline run specs
- creation of run directories
- the full contents of `run_state.json`
- state transitions for `created`, `submitted`, `running`, `completed`, `failed`, `cancelled`
- persistence of `submitted_at`
- persistence of `slurm_job_id`
- Slurm script generation for pipeline runs
- Slurm submission for pipeline runs
- Slurm cancellation for pipeline runs
- machine-readable CLI output contracts

### `alpaca-ui` owns

- API schemas that represent the UI's HTTP surface
- mapping frontend forms to pipeline JSON config payloads
- SSH command execution
- temp JSON upload paths on the HPC
- polling strategy and API pagination
- display-only derived UI state

### `alpaca-ui` must not own

- sidecar metadata for pipeline run lifecycle
- direct writes to `run_state.json`
- submission timestamps for pipeline runs
- Slurm job IDs for pipeline runs
- parsing human-readable stdout as a required control path

## Deployment topology decision

This spec assumes the following deployment topology because it matches the validated current environment:

- `alpaca-pipelines`, `alpaca-audio-standardizer`, and `alpaca-dataset-builder` are installed on the HPC filesystem and run there
- `alpaca-ui` connects over SSH and executes commands remotely
- `alpaca-ui` does not require its own source checkout on the HPC

This is not just a design assumption. It is the validated current deployment state:

- the HPC has the three tool repos
- the HPC does not have an `alpaca-ui` repo checkout at the expected project path

As a result:

- the `.env` file for `alpaca-ui` is a local deployment concern for wherever the backend process runs
- the `.env` files for `alpaca-pipelines`, `alpaca-audio-standardizer`, and `alpaca-dataset-builder` are remote repository concerns on the HPC
- this migration must not assume that changing files in a remote `alpaca-ui` checkout is part of rollout

## Proposed architecture

### 1. Make `alpaca-pipelines` the owner of submission, not just script generation

Add a first-class CLI/API operation:

- `alpaca-pipelines submit --run-id <id> [--slurm-config <json>]`

This operation must:

- generate the Slurm script if needed
- submit it with `sbatch`
- transition the run from `created` to `submitted`
- write `submitted_at` into `run_state.json`
- write `slurm_job_id` into `run_state.json`
- return machine-readable output

After this change, `alpaca-ui` must stop writing `backend_meta.json`.

`submit` must be implemented in the Python API and exposed by the CLI. The CLI must not shell out to another CLI command internally.

### 2. Make pipeline cancellation own Slurm cancellation when possible

`alpaca-pipelines cancel --run-id <id>` should become the canonical cancellation boundary.

If a run has a persisted `slurm_job_id` and is still cancellable, the command must:

- attempt `scancel <job_id>`
- transition the run to `cancelled`
- persist the result in `run_state.json`

This keeps submission and cancellation ownership in one place.

Cancellation semantics are:

- if the run is `created`, mark it `cancelled` without calling Slurm
- if the run is `submitted` and `slurm_job_id` is present, call `scancel` first and only mark `cancelled` if `scancel` succeeds
- if the run is `running`, cancellation support is out of scope for this spec unless current execution semantics already support it safely
- if the run is terminal, fail clearly

### 3. Add machine-readable CLI output for UI-facing commands

Human-oriented stdout is fine for operators, but it must not be the only integration surface for another service.

Add `--json` output to these commands:

- `create`
- `submit`
- `inspect`
- `list`
- `cancel`
- `generate-slurm`
- `export-selection-tables`

Requirements:

- `create --json` returns the full persisted run state
- `submit --json` returns the full updated run state
- `inspect --json` returns the full run state
- `list --json` returns an object with a `runs` array
- `cancel --json` returns the full updated run state
- `generate-slurm --json` returns at least `run_id` and `script_path`
- `export-selection-tables --json` returns the existing summary payload

The plain-text mode can remain for manual use, but JSON output is the only supported service-to-service interface.

`--json` must write exactly one JSON document to stdout and must not mix human log lines into stdout.

When `--json` is used:

- user-facing diagnostics must go to stderr
- process exit status must remain the source of success/failure
- the JSON payload shape must be stable across patch releases unless the contract files are updated in the same change

### 4. Publish machine-readable config contracts from `alpaca-pipelines`

Today `alpaca-ui` mirrors pipeline config models locally. That is acceptable only if the contract is explicitly owned by `alpaca-pipelines`.

`alpaca-pipelines` should publish JSON Schema files for:

- `TrainingRunSpec`
- `PredictionRunSpec`
- `EvaluationRunSpec`
- `RfTrainingRunSpec`
- `SlurmConfig`
- `RunState`

Proposed location:

- `contracts/json-schema/`

These schemas should be generated from the authoritative Pydantic models during development and committed to the repo.

This gives downstream consumers a stable contract without requiring a Python package dependency.

The committed schemas are the compatibility contract for downstream tools. They are not optional build artifacts.

### 5. Treat `run_state.json` as the only pipeline run state file

For pipeline runs, there must be a single authoritative state document:

- `run_state.json`

`alpaca-ui` may read it.
`alpaca-ui` must not augment it through sidecar state files.

If additional run-level metadata is required for pipeline lifecycle, that metadata must be added to `alpaca-pipelines`'s own contract and written by `alpaca-pipelines`.

`backend_meta.json` must be considered deprecated immediately and removed from `alpaca-ui` after the migration lands.

### 6. Add an explicit migration utility for legacy submitted runs

Because the validated HPC state already contains pipeline runs where `backend_meta.json` holds the real submission metadata, `alpaca-pipelines` must ship a one-time migration command.

Required command:

- `alpaca-pipelines migrate-backend-meta [--runs-root <path>] [--json]`

This command must:

- scan all pipeline run directories under `ALPACA_RUNS_ROOT`
- detect runs where `backend_meta.json` exists
- read `submitted_at` and `slurm_job_id` from that file
- patch `run_state.json` only when the corresponding fields are currently null
- never overwrite non-null `run_state.json` values
- report exactly which runs were migrated, skipped, or found inconsistent

After the migration is run successfully, `backend_meta.json` files may be deleted manually or by a separate cleanup command, but deletion is not required for correctness.

### 7. Keep `ALPACA_UI_JOBS_ROOT` separate and unchanged

The validated HPC state shows that `ALPACA_RUNS_ROOT` currently contains both:

- pipeline-owned runs
- `alpaca-ui` remote jobs under `alpaca-ui-jobs/`

That is acceptable for now because the namespaces are distinct.

This migration must not:

- move `alpaca-ui-jobs`
- rename `ALPACA_UI_JOBS_ROOT`
- attempt to fold dataset-builder or standardizer jobs into `alpaca-pipelines`

The only required change is that pipeline submission metadata stops depending on a sidecar file.

## State model requirements

The canonical pipeline state model after this change is:

- `created`
- `submitted`
- `running`
- `completed`
- `failed`
- `cancelled`

Required transition rules:

- `create` writes `created`
- `submit` transitions `created -> submitted`
- `execute` transitions `created -> running` or `submitted -> running`
- executor success transitions `running -> completed`
- executor failure transitions `running -> failed`
- `cancel` transitions `created -> cancelled`
- `cancel` transitions `submitted -> cancelled`

Required field semantics:

- `created_at` is set exactly once at run creation
- `submitted_at` is set exactly once when submission succeeds
- `started_at` is set exactly once when execution begins
- `completed_at` is set exactly once for `completed`, `failed`, or `cancelled`
- `slurm_job_id` is null before submission and immutable after successful submission

`submitted_at` and `slurm_job_id` must never be populated by `alpaca-ui`.

For legacy runs migrated from `backend_meta.json`, the backfilled values become authoritative immediately once written into `run_state.json`.

## Non-goals

- Replacing `alpaca-pipelines` run persistence with `alpaca-ui` `remote_job_service`
- Making `alpaca-pipelines` reuse dataset-builder or standardizer internals
- Introducing a new network service or daemon on the HPC
- Solving frontend form generation from schemas in this change
- Preserving the current `backend_meta.json` design

## User stories / flows

### Flow 1: create a run without submission

1. `alpaca-ui` writes a validated config JSON to a temporary HPC path
2. `alpaca-ui` runs `alpaca-pipelines create <type> --config ... --json`
3. `alpaca-pipelines` validates the config, creates the run, persists `run_state.json`, and returns the full state
4. `alpaca-ui` uses the returned JSON directly

### Flow 2: submit an existing run

1. User clicks submit in `alpaca-ui`
2. `alpaca-ui` writes optional Slurm override JSON to a temporary HPC path
3. `alpaca-ui` runs `alpaca-pipelines submit --run-id ... --slurm-config ... --json`
4. `alpaca-pipelines` generates the script, submits it, records `submitted_at` and `slurm_job_id`, persists `run_state.json`, and returns the updated state
5. `alpaca-ui` immediately shows the run as submitted without synthesizing state

### Flow 3: poll or reload run status

`alpaca-ui` may either:

- continue reading `run_state.json` directly from the HPC, or
- call `alpaca-pipelines inspect --run-id ... --json`

Both must expose the same data contract.

### Flow 4: cancel a submitted run

1. User clicks cancel in `alpaca-ui`
2. `alpaca-ui` runs `alpaca-pipelines cancel --run-id ... --json`
3. `alpaca-pipelines` cancels the Slurm job when applicable, updates `run_state.json`, and returns the new state

### Flow 5: migrate existing HPC runs after rollout

1. Operator deploys the new `alpaca-pipelines` version to the HPC repo and refreshes its venv
2. Operator runs `alpaca-pipelines migrate-backend-meta --json`
3. The command scans current run directories and backfills `submitted_at` and `slurm_job_id` into legacy `run_state.json` files
4. `alpaca-ui` can then stop reading `backend_meta.json` without breaking existing run detail pages

## API / CLI contract

### New command

`alpaca-pipelines submit --run-id <id> [--slurm-config <path>] [--json]`

Behavior:

- fails if run does not exist
- fails if run is not in `created`
- generates a Slurm script in the run's `slurm/` directory
- submits the job
- records `submitted` state in `run_state.json`

### Updated command requirements

`alpaca-pipelines create <type> --config <path> [--json]`

- validates the config against the authoritative run spec model
- creates the run directory scaffold
- persists the initial `run_state.json`
- returns the full run state in JSON mode

`alpaca-pipelines inspect --run-id <id> [--json]`

- returns the persisted state for exactly one run

`alpaca-pipelines list [--type ...] [--status ...] [--json]`

- returns all readable runs matching filters
- does not fail the whole command because one run directory is corrupt
- in JSON mode, corrupt runs may be omitted but must emit a warning to stderr

`alpaca-pipelines cancel --run-id <id> [--json]`

- performs the canonical cancellation behavior defined above
- returns the updated persisted run state in JSON mode

`alpaca-pipelines generate-slurm --run-id <id> [--slurm-config <path>] [--json]`

- remains available as a lower-level operator command
- must not change lifecycle state by itself
- must not write `submitted_at`
- must not write `slurm_job_id`

`alpaca-pipelines migrate-backend-meta [--runs-root <path>] [--json]`

- is idempotent
- does not require `alpaca-ui`
- only mutates legacy runs whose submission fields are still absent in `run_state.json`
- fails non-zero if it encounters an internally inconsistent run that cannot be migrated safely

### JSON output shape

For commands that return a run:

```json
{
  "run_id": "uuid",
  "run_type": "training",
  "status": "submitted",
  "created_at": "2026-03-09T10:00:00Z",
  "submitted_at": "2026-03-09T10:05:00Z",
  "started_at": null,
  "completed_at": null,
  "spec": {},
  "outputs": {},
  "progress": {},
  "error_message": null,
  "slurm_job_id": "12345",
  "run_dir": "/path/to/run"
}
```

For `list --json`:

```json
{
  "runs": []
}
```

For `generate-slurm --json`:

```json
{
  "run_id": "uuid",
  "script_path": "/path/to/job.sbatch"
}
```

### Error contract

When a command fails in JSON mode:

- exit status must be non-zero
- stdout must be empty
- stderr must contain a single-line human-readable error message

This spec deliberately does not require JSON-formatted stderr errors. `alpaca-ui` should use exit status plus stderr text for diagnostics.

The migration command is the one exception for bulk operations:

- in JSON mode it may emit a single summary object to stdout on success
- if it fails, stdout must still be empty and stderr must describe the blocking inconsistency

### Slurm submission contract

Submission must use the same Slurm configuration semantics as current `generate-slurm`.

If `--slurm-config` is provided:

- the file must be validated against `SlurmConfig`
- invalid config must fail before any state mutation

If submission succeeds:

- the returned `slurm_job_id` must be the exact scheduler job ID returned by `sbatch`
- the generated script path must be deterministic: `<run_dir>/slurm/job.sbatch`

If submission fails:

- `run_state.json` must remain in `created`
- `submitted_at` must remain null
- `slurm_job_id` must remain null

## Environment and wiring requirements

### `alpaca-ui` backend environment

The backend process running `alpaca-ui` must be configured with:

- SSH connectivity to the HPC
- the same remote persistence paths that have been validated on the HPC

Required variables in the `alpaca-ui` backend environment:

- `HPC_HOSTNAME`
- `HPC_USERNAME`
- `HPC_KEY_PATH` or agent-based SSH configuration
- `ALPACA_COLLECTION_ROOT`
- `ALPACA_MERGED_INDEX`
- `ALPACA_DATASETS_ROOT`
- `ALPACA_RUNS_ROOT`
- `ALPACA_UI_JOBS_ROOT`
- `ALPACA_IDENTITY_MAP_PATH`
- `ALPACA_STANDARDIZER_ARTIFACTS_ROOT`
- `ALPACA_STANDARDIZER_VENV`
- `ALPACA_STANDARDIZER_COMMAND`
- `ALPACA_DATASET_BUILDER_VENV`
- `ALPACA_DATASET_BUILDER_COMMAND`
- `SLURM_DEFAULT_VENV`

For the validated deployment, the correct remote values are:

- `ALPACA_COLLECTION_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection`
- `ALPACA_MERGED_INDEX=/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json`
- `ALPACA_DATASETS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/datasets`
- `ALPACA_RUNS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/runs`
- `ALPACA_UI_JOBS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/runs/alpaca-ui-jobs`
- `ALPACA_IDENTITY_MAP_PATH=/projects/extern/kisski/kisski-alpaca-2/dir.project/config/identity_map.json`
- `ALPACA_STANDARDIZER_ARTIFACTS_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/data`
- `ALPACA_STANDARDIZER_VENV=/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-audio-standardizer/.venv`
- `ALPACA_DATASET_BUILDER_VENV=/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-dataset-builder/.venv`
- `SLURM_DEFAULT_VENV=/projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-pipelines/.venv`

No new `alpaca-ui` backend env variable is required for this migration.

The only backend config change required after rollout is behavioral:

- stop treating `backend_meta.json` as part of the pipeline state contract

### `alpaca-pipelines` remote environment

The remote `alpaca-pipelines` repository already has `.env` and `.env.example` files on the HPC.

This migration requires that the remote `alpaca-pipelines` `.env` continue to define:

- `ALPACA_COLLECTION_ROOT`
- `ALPACA_MERGED_INDEX`
- `ALPACA_DATASETS_ROOT`
- `ALPACA_RUNS_ROOT`

No new mandatory environment variable is required to implement:

- `submit`
- `--json`
- `migrate-backend-meta`

because the validated HPC environment already provides:

- the runs root
- the datasets root
- the collection root
- Slurm client commands

### `alpaca-audio-standardizer` and `alpaca-dataset-builder` remote `.env`

The checked HPC `.env` files confirm:

- `alpaca-dataset-builder/.env` is aligned with the shared persistence root used by the UI
- `alpaca-audio-standardizer/.env` is not aligned with the UI's current production integration boundary because it points to a work-copy path and a repo-relative identity map

That is acceptable because `alpaca-ui` does not rely on those `.env` files for orchestration. It activates the remote venvs and passes explicit command arguments and exported environment values.

Therefore this migration must preserve that rule:

- `alpaca-ui` must continue to treat the remote standardizer and dataset-builder repos as executable tools, not as configuration authorities

The authoritative remote wiring for the UI remains the backend's own configured HPC paths, not the checked-in `.env` of sibling repos.

### `.env.example` updates required

`alpaca-pipelines/.env.example` should be updated to describe:

- that `ALPACA_RUNS_ROOT` is both the persistence root and the location scanned by `migrate-backend-meta`
- that legacy `backend_meta.json` files may exist during migration
- that `submit` persists submission metadata into `run_state.json`

`alpaca-ui/.env.example` should be updated to describe:

- that `ALPACA_RUNS_ROOT` contains the authoritative pipeline `run_state.json` files
- that `ALPACA_UI_JOBS_ROOT` is only for non-pipeline UI-managed jobs
- that `backend_meta.json` is no longer part of the supported pipeline contract after migration
- that `alpaca-ui` runs remotely over SSH and is not expected to exist as a checkout on the HPC

No `.env.example` change is required in the standardizer or dataset-builder repos for this migration.

## Manual HPC preflight and rollout procedure

These are required operator checks and commands for a trustworthy rollout.

All commands must be run over SSH using the `gwdg-kisski` alias.

### Preflight checks

1. Verify connectivity:
   - `ssh gwdg-kisski 'pwd; uname -n'`
2. Verify persistence roots exist:
   - `ssh gwdg-kisski 'ls -ld /projects/extern/kisski/kisski-alpaca-2/dir.project/{config,data,datasets,runs}'`
3. Verify input files exist:
   - `ssh gwdg-kisski 'ls -l /projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json /projects/extern/kisski/kisski-alpaca-2/dir.project/config/identity_map.json'`
4. Verify installed entrypoints exist:
   - `ssh gwdg-kisski 'ls -l /projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-pipelines/.venv/bin/alpaca-pipelines /projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-audio-standardizer/.venv/bin/alpaca-audio /projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-dataset-builder/.venv/bin/alpaca-dataset'`
5. Verify legacy sidecar metadata count before migration:
   - `ssh gwdg-kisski 'find /projects/extern/kisski/kisski-alpaca-2/dir.project/runs -name backend_meta.json | sort'`

### Deploy and migrate

1. Deploy the updated `alpaca-pipelines` code to the remote HPC repo.
2. Refresh the remote pipeline venv so the new CLI is installed.
3. Run the migration:
   - `ssh gwdg-kisski 'bash -lc \"source /projects/extern/kisski/kisski-alpaca-2/dir.project/alpaca-pipelines/.venv/bin/activate && alpaca-pipelines migrate-backend-meta --json\"'`
4. Verify no run still depends on sidecar-only submission state.
   - Acceptance rule:
     for every run that has a Slurm log file or had a legacy `backend_meta.json`, `run_state.json` must now carry the same `submitted_at` and `slurm_job_id`
5. Deploy the updated `alpaca-ui` backend config and code.
6. Verify the UI can load existing runs without reading `backend_meta.json`.

### Cleanup

Cleanup is optional and must happen only after the new UI is confirmed healthy.

Optional cleanup command pattern:

- archive or delete legacy `backend_meta.json` files after migration verification

This spec does not require automated deletion.

## Acceptance criteria

- `alpaca-ui` can remove `alpaca-pipelines` from backend Python dependencies
- `alpaca-ui` no longer parses human stdout to discover `run_id`
- `alpaca-ui` no longer writes `backend_meta.json`
- `submitted_at` and `slurm_job_id` are owned by `alpaca-pipelines` and persisted in `run_state.json`
- `cancel` is able to cancel submitted Slurm jobs through the pipeline-owned boundary
- `alpaca-pipelines` publishes committed JSON Schema files for its UI-facing contracts
- a downstream consumer can integrate with `alpaca-pipelines` using only SSH, the CLI, and documented file contracts
- `generate-slurm` remains available for manual and debugging workflows without becoming the primary UI submission path
- JSON mode outputs are sufficient for `alpaca-ui` to stop regex-parsing CLI output
- the persisted state read by `inspect --json` matches the raw `run_state.json` contents
- no pipeline lifecycle field required by the UI depends on any sidecar file
- the rollout instructions are sufficient to validate the current HPC persistence layer before changing code
- the migration plan covers already-existing legacy runs on the HPC
- the environment section makes clear that no remote `alpaca-ui` checkout is assumed

## Edge cases

- `submit` after already submitted/running/completed/cancelled: fail with a clear error
- `submit` where `sbatch` fails: do not transition to `submitted`
- `cancel` where `scancel` reports missing job but run is still `submitted`: fail clearly rather than silently synthesizing success
- `create --json` with invalid config: fail with non-zero exit and machine-readable stderr message
- `inspect --json` for missing run: non-zero exit
- schema changes in any run spec: the corresponding JSON Schema file must be updated in the same change
- a partially written or corrupt `run_state.json` in one run directory must not break `list --json` for all other runs
- `submit` must not overwrite an existing `slurm_job_id`
- `submit` must fail if the generated script path cannot be written
- `cancel` on a `created` run with no Slurm job must still work
- `cancel` on a run with inconsistent state (`submitted` but no `slurm_job_id`) must fail loudly because that indicates contract corruption
- `migrate-backend-meta` must skip runs that have no sidecar metadata
- `migrate-backend-meta` must skip runs whose `run_state.json` already has non-null `submitted_at` and `slurm_job_id`
- `migrate-backend-meta` must fail if sidecar metadata conflicts with non-null values already present in `run_state.json`
- `migrate-backend-meta` must tolerate `alpaca-ui-jobs/` being present under the same runs root without attempting to interpret those entries as pipeline runs

## Required tests

`alpaca-pipelines` must add tests for:

- `create --json` returns parseable JSON for each run type
- `submit --json` persists `submitted_at` and `slurm_job_id`
- failed `sbatch` leaves state unchanged
- `cancel` on `created` transitions to `cancelled`
- `cancel` on `submitted` calls `scancel` and persists `cancelled`
- `generate-slurm --json` does not mutate lifecycle state
- JSON Schema artifacts are generated from the current models and are kept in sync
- `migrate-backend-meta` backfills legacy runs correctly
- `migrate-backend-meta` is idempotent
- `migrate-backend-meta` ignores `alpaca-ui-jobs`

`alpaca-ui` must add or update tests for:

- pipeline creation no longer depends on parsing stdout text
- pipeline submission no longer writes or reads `backend_meta.json`
- run detail and dashboard views read `submitted_at` and `slurm_job_id` from the pipeline-owned state only
- existing migrated runs still render correctly after reload

## Migration plan

### Phase 1: add pipeline capabilities

- implement `PipelineAPI.submit_run(...)`
- implement CLI `submit`
- add `--json` support to the required commands
- implement `migrate-backend-meta`
- add schema export artifacts
- add tests in `alpaca-pipelines`

### Phase 2: switch `alpaca-ui`

- update `PipelineService` to call `create --json`
- update `PipelineService` to call `submit --json`
- remove `backend_meta.json` reads and writes
- remove stdout regex parsing for `run_id`
- keep direct `run_state.json` reads only if they continue to match the contract exactly
- do not require any remote `alpaca-ui` filesystem changes on the HPC

### Phase 3: remove dependency

- remove `alpaca-pipelines` from `packages/backend/pyproject.toml`
- keep local request/response schemas in `alpaca-ui`
- document that pipeline contracts come from committed JSON Schema, not Python imports

## Migration / rollout notes

Implementation should happen in this order:

1. Add `submit` to `alpaca-pipelines` API and CLI.
2. Move `submitted_at` and `slurm_job_id` persistence into `run_state.json`.
3. Add `migrate-backend-meta` for existing HPC runs.
4. Add `--json` output to the UI-facing commands.
5. Publish JSON Schema files under `contracts/json-schema/`.
6. Run the HPC preflight checks and migrate existing runs.
7. Update `alpaca-ui` to use `create --json` and `submit --json`.
8. Remove `backend_meta.json` reads and writes from `alpaca-ui`.
9. Remove the `alpaca-pipelines` dependency from `alpaca-ui` backend.

No compatibility layer is required. This is a coordinated change across the two repos and should land as one migration sequence rather than preserving the old sidecar metadata path.

## Explicit rejection

This spec intentionally rejects one tempting direction:

- making `alpaca-pipelines` look like `remote_job_service`

That would align implementation shape but break ownership. The cleaner architecture is:

- `alpaca-ui` owns durable jobs for tools that do not have their own durable lifecycle
- `alpaca-pipelines` owns its own durable lifecycle
- all three tools expose clean remote execution boundaries and explicit contracts
