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

### 5.4 Prediction outputs contract (required)

This section specifies the output artifacts written under:

`ALPACA_RUNS_ROOT/prediction/<run_id>/outputs/predictions/`

Hard rules:

* Prediction outputs MUST be written only under the run directory for that run.
* The system MUST fail immediately if any required file is missing or malformed.
* The system MUST NOT guess or infer files by scanning directories. It MUST use explicit pointers / summaries described below.

#### 5.4.1 Per-file prediction JSON

For each predicted audio file, `alpaca-pipelines` MUST write one JSON file named:

* `<audio_stem>.json`

Where `<audio_stem>` is `Path(audio_file).stem` computed from the `audio_file` string that was predicted.

Hard rules:

* `<audio_stem>` MUST be derived as a basename stem; it MUST NOT contain path separators.
* If two audio files produce the same `<audio_stem>`, the system MUST fail immediately (filename collision).
* The system MUST fail immediately if the output path already exists.

Top-level JSON object for `<audio_stem>.json`:

* `audio_file`: string (the exact audio file path used for prediction; may be absolute)
* `n_windows`: int
* `n_detections`: int
* `detections`: array of objects, each with EXACT fields:

  * `start_s`: float
  * `end_s`: float
  * `score`: float
* `scores_shape`: array of exactly two ints: `[n_windows, n_classes]`

Hard rules:

* `n_detections` MUST equal the length of `detections`.
* Times are in seconds and must be non-negative; malformed values MUST fail.

#### 5.4.2 Prediction summary JSON (required)

After producing all per-file outputs, `alpaca-pipelines` MUST write:

* `prediction_summary.json`

at:

`ALPACA_RUNS_ROOT/prediction/<run_id>/outputs/predictions/prediction_summary.json`

Top-level JSON object for `prediction_summary.json`:

* `run_id`: string
* `model_path`: string (path to the model checkpoint used; may be absolute)
* `n_files`: int
* `total_detections`: int
* `detection_threshold`: float
* `files`: array of objects, each with EXACT fields:

  * `audio_file`: string
  * `n_windows`: int
  * `n_detections`: int

Hard rules:

* `n_files` MUST equal the length of `files`.
* `total_detections` MUST equal the sum of `n_detections` across `files`.
* `files[].audio_file` MUST be the authoritative list of predicted audio files for downstream post-processing.

#### 5.4.3 RF-filtered per-file prediction JSON (optional)

If RF post-processing is applied, for each predicted audio file the system MUST write a separate JSON file named:

* `<audio_stem>_rf_filtered.json`

Top-level JSON object MUST include all fields from the corresponding non-RF file, and additionally:

* `rf_filtered`: bool (must be `true`)
* `rf_model_path`: string (path to RF model used; may be absolute)

Each detection object MAY additionally include:

* `rf_score`: float or null
* `rf_pass`: bool

Hard rules:

* RF-filtered outputs MUST NOT overwrite the base `<audio_stem>.json`.
* If RF-filtered output is requested by downstream consumers but the file is missing, the system MUST fail immediately.

---

### 5.5 Prediction selection table outputs (post-processing) (required)

This section specifies the persisted Raven selection tables derived from a **completed** prediction run.

These artifacts MUST be stored under the prediction run directory and MUST be reproducible from the prediction outputs.

#### 5.5.1 Output location (required)

Selection tables MUST be written under:

`ALPACA_RUNS_ROOT/prediction/<run_id>/outputs/predictions/selection_tables/`

Directory layout:

```
ALPACA_RUNS_ROOT/
└── prediction/
    └── <run_id>/
        └── outputs/
            └── predictions/
                └── selection_tables/
                    ├── selection_tables_summary.json
                    ├── <audio_stem>.txt
                    └── <audio_stem>_rf_filtered.txt   (only for RF-filtered export)
```

Hard rules:

* Post-processing MUST NOT write outside the run directory.
* The system MUST fail immediately if `prediction_summary.json` is missing or malformed.
* The system MUST NOT infer audio files by scanning directories. It MUST use `prediction_summary.json` as the authoritative list.
* The system MUST fail immediately if any required per-file prediction JSON is missing.

#### 5.5.2 Export mode selection (hard)

Selection tables can be exported in exactly one of two modes:

* **Base mode**: uses `<audio_stem>.json`
* **RF-filtered mode**: uses `<audio_stem>_rf_filtered.json`

Hard rules:

* In RF-filtered mode, the system MUST read `<audio_stem>_rf_filtered.json` and MUST fail immediately if it does not exist.
* In base mode, the system MUST read `<audio_stem>.json` and MUST fail immediately if it does not exist.
* The mode MUST be explicit (no auto-detection).

#### 5.5.3 Raven selection table TSV contract (required)

Each selection table file is a tab-separated file (`.txt`) compatible with Raven Pro.

The TSV MUST include a header row with EXACT columns in EXACT order:

1. `Selection`
2. `View`
3. `Channel`
4. `Begin Time (s)`
5. `End Time (s)`
6. `Low Freq (Hz)`
7. `High Freq (Hz)`
8. `Score`

Hard rules:

* The header row MUST be present even if there are zero detections.
* Each data row MUST correspond to exactly one exported detection.
* `Selection` MUST be 1-indexed and strictly increasing by 1 per row.
* `Begin Time (s)` and `End Time (s)` MUST be copied from detection `start_s` / `end_s`.
* `Score` MUST be:

  * base mode: detection `score`
  * RF-filtered mode: detection `rf_score` (may be null)
* The exporter MUST NOT reorder detections unless explicitly specified by the pipeline logic.

Filtering rule for RF-filtered mode:

* If `rf_pass` exists and is `false`, that detection MUST be excluded from the selection table.
* If `rf_pass` is missing, it MUST be treated as `true` (i.e., included).

`Low Freq (Hz)` and `High Freq (Hz)`:

* MUST be set from explicit parameters provided by Makefile/API invocation.
* No defaults are allowed when the caller claims these are required inputs.
* If provided, they MUST be integers and `0 <= Low < High`.

#### 5.5.4 Selection tables summary JSON contract (required)

The system MUST write:

* `selection_tables_summary.json`

at:

`ALPACA_RUNS_ROOT/prediction/<run_id>/outputs/predictions/selection_tables/selection_tables_summary.json`

Top-level JSON object with EXACT fields:

* `generated_at`: string (ISO8601 Z)
* `predictions_dir`: string (absolute path)
* `selection_tables_dir`: string (absolute path)
* `use_rf_filtered`: bool
* `freq_low_hz`: int
* `freq_high_hz`: int
* `n_files`: int
* `files`: array of objects, each with EXACT fields:

  * `audio_file`: string (exact value from `prediction_summary.json`)
  * `audio_file_stem`: string
  * `predictions_json`: string (absolute path to the per-file prediction JSON used)
  * `selection_table`: string (absolute path to the TSV written)

Hard rules:

* `n_files` MUST equal the length of `files`.
* The `files` array MUST have exactly one entry per `prediction_summary.json` file entry.
* The system MUST fail immediately if any output `.txt` path already exists.

---

## 6) Contract reminders (implementation constraints)

* `alpaca-pipelines` MUST NOT import or couple to `alpaca-dataset-builder` or `alpaca-audio-standardizer`.
* It reads `manifest.json`, `splits/*.csv`, and `merged_index.json` as stable public interfaces.
* API-first design: all operations are available as programmatic methods on `PipelineAPI`. The CLI is a thin wrapper around the API. The future backend drives the API directly.
