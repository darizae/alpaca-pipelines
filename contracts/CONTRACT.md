# Persistence Layer Contract (Defaults)

This document specifies the **folder-based persistence layer** used by `alpaca-pipelines` and the future React/FastAPI app.

It is a **hard contract**: implement exactly what is described here. Do not invent fields. Do not treat root-relative paths as absolute. If required inputs are missing or malformed, the system **must fail immediately** with clear errors.

---

## 1) Core principles (hard rules)

### 1.1 Folder-based persistence only (no SQL)
- All pipeline state is stored as JSON files under `ALPACA_RUNS_ROOT`.
- Each run occupies a single directory containing its specification/state, log pointers, and output pointers.

### 1.2 No embedded paths
`alpaca-pipelines` MUST be configured only by explicit inputs:
- Environment variables for Makefile usage, and
- Explicit parameters for API usage.

It MUST NOT embed any user-specific or cluster-specific absolute paths.

### 1.3 No fallback mechanisms
If a required input is missing or malformed, the system MUST fail immediately with a clear error.
- No silent salvage
- No defaults for required fields

### 1.4 Path safety (mandatory validation)
Any relative path read from JSON or CSV MUST be validated:
- no absolute paths
- no `..`
- final resolved path must remain inside its root

If validation fails, the system MUST fail immediately with a clear error.

### 1.5 Run isolation
- Each run is identified by a UUID and stored in its own directory.
- Runs MUST NOT mutate each other's state.
- Concurrent runs are safe because they operate on separate directories.

---

## 2) Required environment variables and defaults (exact paths)

For Makefile execution, `alpaca-pipelines` MUST support reading a local `.env` file (same pattern as the existing repos) and expose these required environment variables:

### 2.1 Required env vars

- `ALPACA_COLLECTION_ROOT`
  Absolute path to the persistence root containing `audio_collection_*` directories.

- `ALPACA_MERGED_INDEX`
  Absolute path to `merged_index.json` under the collection root.

- `ALPACA_DATASETS_ROOT`
  Absolute path to datasets root where dataset directories (`strategy_name`) exist or will be created.

- `ALPACA_RUNS_ROOT`
  Absolute path to a runs/work root where `alpaca-pipelines` writes its own orchestration artifacts
  (job specs, job state/status JSON, logs pointers, outputs pointers). This is required so a backend
  can poll status and fetch results deterministically.

  This root is NOT the audio persistence root; it is `alpaca-pipelines`’ own state/output root.

### 2.2 Default values (examples only; these are defaults)

These paths are defaults:

- `ALPACA_COLLECTION_ROOT`:
  `/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection`

- Example collections under `ALPACA_COLLECTION_ROOT`:
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/audio_collection_1`
  - `/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/audio_collection_2`

- `ALPACA_MERGED_INDEX`:
  `/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json`

- `ALPACA_DATASETS_ROOT`:
  `/projects/extern/kisski/kisski-alpaca-2/dir.project/datasets`

- Example dataset directory (under `ALPACA_DATASETS_ROOT`):
  `/projects/extern/kisski/kisski-alpaca-2/dir.project/datasets/clipwise_balanced_nph_2/`
  - `snippets/`
  - `splits/train.csv`
  - `manifest.json`

---

## 3) Path resolution rules (hard)

### 3.1 `merged_index.json` entry paths
Whenever the system reads `hum_path` from any index entry, it MUST treat it as a **root-relative POSIX path** and resolve it using:

- `Path(ALPACA_COLLECTION_ROOT) / hum_path`

Hard rule:
- `hum_path` MUST be resolved as `Path(ALPACA_COLLECTION_ROOT) / hum_path` (after validating no absolute/traversal).

Contract reminder:
- `merged_index.json` entries contain `hum_path` like:
  `audio_collection_1/hums_segmented/<filename>.wav`
  which must resolve as:
  `Path(ALPACA_COLLECTION_ROOT) / hum_path`

### 3.2 Relative path validation for JSON/CSV inputs
Any relative path read from JSON or CSV MUST be validated:
- Must NOT be absolute
- Must NOT contain `..`
- Final resolved path MUST remain inside its root

### 3.3 `alpaca-pipelines` agnosticism (hard)
- The backend or Makefile provides these roots.
- `alpaca-pipelines` only validates and uses them.
- If any required env var / parameter is missing, `alpaca-pipelines` MUST fail immediately with a clear error listing the missing keys.

---

## 4) Read-only input contracts (JSON + CSV)

You MUST implement reading/writing these JSON artifacts EXACTLY as specified.
- Do not invent fields.
- Do not treat root-relative paths as absolute.

### 4.1 A) `merged_index.json` contract

**Purpose:** input to dataset building and potentially to `alpaca-pipelines` evaluation / training selection.

Top-level JSON object:

- `meta`: object
  - `generated_at`: string (ISO8601 Z) (may exist)
  - `n_collections`: int
  - `n_total_hums`: int
- `entries`: array of objects, each with EXACT fields:
  - `collection`: string (e.g., `"audio_collection_1"`)
  - `subject_id`: string
  - `recording_date`: `"YYYY-MM-DD"`
  - `recording_time`: `"HH:MM:SS"` or null
  - `hum_path`: string ROOT-RELATIVE to `ALPACA_COLLECTION_ROOT` INCLUDING the collection dir prefix
    (e.g., `"audio_collection_1/hums_segmented/....wav"`)
  - `hum_start_s`: float
  - `hum_end_s`: float
  - `source_quality`: int
  - `keep`: bool
  - `hum_uid`: int

Example (verbatim shape; values are illustrative but consistent with the contract):

```json
{
  "meta": {
    "generated_at": "2026-03-01T16:23:03Z",
    "n_collections": 2,
    "n_total_hums": 1730
  },
  "entries": [
    {
      "collection": "audio_collection_1",
      "subject_id": "387",
      "recording_date": "2020-12-07",
      "recording_time": null,
      "hum_path": "audio_collection_1/hums_segmented/387_20201207_cut.wav_110_125.wav_450.653258954_451.035708526Q3.wav",
      "hum_start_s": 450.653258954,
      "hum_end_s": 451.035708526,
      "source_quality": 3,
      "keep": true,
      "hum_uid": 1
    }
  ]
}
````

Hard rule:

* `hum_path` MUST be resolved as `Path(ALPACA_COLLECTION_ROOT) / hum_path` (after validating no absolute/traversal).

---

### 4.2 B) Dataset `manifest.json` contract

**Purpose:** input/output for dataset usage; `alpaca-pipelines` must be able to read it.

Top-level JSON object:

* `meta`: object with EXACT fields:

  * `strategy_name`: string
  * `created_at`: string (ISO8601 Z)
  * `collection_root`: string (absolute path used when built; do not assume current env matches; validate when used)
  * `merged_index_path`: string (absolute path used when built; do not assume current env matches; validate when used)
  * `seed`: int
  * `n_snippets`: int
  * `n_target`: int
  * `n_noise`: int
  * `manifest_hash`: string
  * `strategy_config`: object or null (if present, contains the dataset config used)

* `snippets`: array of objects with EXACT fields:

  * `uid`: int
  * `filename`: string (basename like `"noise-bg_001858_audio_collection_2.wav"`) stored under `<dataset_dir>/snippets/`
  * `classification`: `"target"` | `"noise"`
  * `source_type`: `"hum"` | `"mined_source"` | `"low_quality_hum"`
  * `source_path`: string ROOT-RELATIVE to collection root (same scheme as `hum_path`, e.g. `"audio_collection_2/clips_labelled/....wav"` or `"audio_collection_1/hums_segmented/....wav"`)
  * `start_s`: float
  * `end_s`: float
  * `duration_s`: float
  * `quality`: int or null
  * `subject_id`: string or null
  * `recording_date`: `"YYYY-MM-DD"` or null
  * `collection`: string
  * `session_key`: string or null
  * `recording_time`: string or null
  * `split`: `"train"` | `"val"` | `"test"` or null
  * `review_status`: `"pending"` | `"approved"` | `"rejected"`

Example (verbatim shape; values from the provided snippet):

```json
{
  "meta": {
    "strategy_name": "clipwise_balanced_nph_2",
    "created_at": "2026-03-01T16:25:18Z",
    "collection_root": "/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection",
    "merged_index_path": "/projects/extern/kisski/kisski-alpaca-2/dir.project/data/raw_audio_collection/merged_index.json",
    "seed": 42,
    "n_snippets": 4557,
    "n_target": 1519,
    "n_noise": 3038,
    "manifest_hash": "997be7ab601c8584301de9d21cc7ab59b2a9e58f02a2a784f9602bdfde8f32d3",
    "strategy_config": {
      "split_strategy": "clipwise_balanced",
      "seed": 42,
      "min_quality": 2,
      "noise_per_positive": 2.0,
      "noise_mining": {
        "attempts_per_slot": 20,
        "source_category_dirs": ["clips_labelled"],
        "low_quality_as_negative": false,
        "low_quality_threshold": 1
      },
      "split_fractions": [0.7, 0.15, 0.15],
      "duration_tolerance_s": 0.1,
      "review_gap_s": 0.5,
      "freq_low_hz": 0,
      "freq_high_hz": 4000
    }
  },
  "snippets": [
    {
      "uid": 1858,
      "filename": "noise-bg_001858_audio_collection_2.wav",
      "classification": "noise",
      "source_type": "mined_source",
      "source_path": "audio_collection_2/clips_labelled/4212_20250205_193000.wav_0_3604.wav",
      "start_s": 2748.864,
      "end_s": 2748.9171,
      "duration_s": 0.0531,
      "quality": null,
      "subject_id": null,
      "recording_date": null,
      "collection": "audio_collection_2",
      "session_key": null,
      "recording_time": null,
      "split": "train",
      "review_status": "approved"
    }
  ]
}
```

Hard rules:

* `source_path` must be treated as ROOT-RELATIVE to a collection root (either `manifest.meta.collection_root` or `ALPACA_COLLECTION_ROOT` depending on how `alpaca-pipelines` is configured).
* Validate no absolute/traversal, and resolve as `Path(collection_root) / source_path`.

---

### 4.3 C) Splits CSV contract

* Splits are stored under: `<dataset_dir>/splits/{train,val,test}.csv`
* Each line contains exactly one snippet filename (the `filename` field from `manifest.snippets`)

Example (conceptual):

```text
noise-bg_001858_audio_collection_2.wav
target-Q3_000001_audio_collection_1.wav
```

Hard rules:

* Filenames are basenames only (no path separators), no traversal.
* Resolve snippet wav as: `<dataset_dir>/snippets/<filename>`

---

## 5) Run state persistence (under `ALPACA_RUNS_ROOT`)

### 5.1 Run directory layout (required)

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

### 5.2 Run state machine

```
created → submitted → running → completed
                             → failed
         → cancelled (from created or submitted)
```

### 5.3 Immutable run specifications (hard)

Once a run is created, its specification MUST NOT change.

* Status, timestamps, and result pointers are the only mutable fields.

---

## 6) Contract reminders (implementation constraints)

* `alpaca-pipelines` MUST NOT import or couple to `alpaca-dataset-builder` or `alpaca-audio-standardizer`.
* It reads `manifest.json`, `splits/*.csv`, and `merged_index.json` as stable public interfaces.
* API-first design: all operations are available as programmatic methods on `PipelineAPI`. The CLI is a thin wrapper around the API. The future backend drives the API directly.
