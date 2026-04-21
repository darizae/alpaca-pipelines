# alpaca-pipelines — Curated Manual-Review Dataset Source Implementation Spec

## Goal

Extend `alpaca-pipelines` so that manually reviewed prediction outputs can become a first-class dataset source for dataset construction, with durable provenance back to the original source recording, prediction run, and review item.

The pipeline must support this workflow:

1. prediction is run on a source recording, often under `raw_recordings/`
2. manual review accepts some predicted segments
3. accepted reviewed segments are assigned curated labels, initially `target` or `noise`
4. those curated segments are materialized as reusable pipeline-side curated examples
5. dataset build can ingest curated examples as valid positive/negative sources
6. built dataset manifests expose provenance showing:
   - which snippets came from manual curation
   - which tape/collection they came from
   - which prediction run/review session they came from
   - how many dataset examples came from manual curation vs pre-existing indexed sources

This spec covers `alpaca-pipelines` only. UI/catalog work is specified separately.

---

## Current-state summary

The pipeline already owns:

- prediction execution
- prediction manual-review artifacts
- dataset building
- dataset review workflows
- filesystem persistence contracts :contentReference[oaicite:4]{index=4}

It also already supports:

- raw-only collections as valid inference sources
- prediction on selected collection/category directories
- provenance-oriented metadata such as `source_recording_key` joins in `recordings.json`, `merged_index.json`, and dataset manifests :contentReference[oaicite:5]{index=5}

### Current limitation

The README states:

- dataset build still derives positive targets from hum index entries only
- raw audio remains a negative/noise source
- if no labelled targets are available, `dataset-build` fails explicitly :contentReference[oaicite:6]{index=6}

That means manual-review-curated prediction segments are not yet a first-class dataset source, especially not for positive targets.

---

## Design principles

1. **Pipeline owns runtime artifacts**
   - curated examples used by dataset build must exist as pipeline-owned filesystem artifacts/contracts

2. **Manual curation becomes a first-class source type**
   - not an ad hoc postprocessing sidecar
   - not only UI metadata

3. **Provenance must survive into dataset manifests**
   - downstream inspection must not require database access only

4. **Existing sources remain supported**
   - indexed hums/clips continue to work
   - curated manual-review examples are additive

5. **The source recording remains authoritative**
   - every curated example should preserve lineage to the original source recording/tape

---

## Scope

### In scope

- curated example filesystem contract
- new CLI/API contracts for curated example export/materialization
- dataset-builder ingestion of curated examples
- dataset manifest provenance schema additions
- dataset summary rollups for provenance

### Out of scope

- UI database/catalog changes
- front-end review workflow
- user-facing label editing UI

---

## Required new concepts

## 1. Curated example source type

Introduce a new dataset-source/provenance type:

- `manual_review_curated`

This source type represents snippets derived from:
- prediction outputs
- reviewed detections
- explicit human curation label assignment

### Initial supported labels

- `target`
- `noise`

---

## 2. Curated example corpus contract

Add a pipeline-owned persistence format for curated examples.

### Recommendation

Create a curated corpus under a stable root, for example:

`ALPACA_COLLECTION_ROOT/<collection>/curated_examples/`
or
`ALPACA_RUNS_ROOT/curated_examples/`
or
`ALPACA_DATASETS_ROOT/_curated_sources/`

### Preferred recommendation

Use a dedicated curated-source root under datasets or a pipeline-owned source root, not inside ephemeral review-session directories.

Example:

`ALPACA_DATASETS_ROOT/_curated_prediction_examples/<collection>/<source_recording_key>/...`

### Required durability property

Curated examples must outlive prediction review sessions and be reusable across multiple dataset builds.

---

## Curated corpus structure

## 1. Required manifest

Each curated source batch or corpus unit must include a manifest JSON.

### Suggested top-level fields

- `schema_version`
- `source_type` = `manual_review_curated`
- `collection_name`
- `source_category_dir`
- `source_relative_path`
- `source_recording_key` if available
- `source_audio_file`
- `prediction_run_id`
- `review_session_id`
- `created_at`
- `items`

### Per-item fields

- `curated_example_id`
- `review_item_id`
- `detection_index`
- `start_s`
- `end_s`
- `duration_s`
- `detection_score`
- `label` (`target` / `noise`)
- `snippet_wav_path`
- `source_recording_key`
- `source_collection_name`
- `source_category_dir`
- `source_relative_path`
- `source_audio_file`
- `prediction_run_id`
- `review_session_id`
- `provenance_type` = `manual_review_curated`
- `payload_json` optional

### Important

The manifest must be self-sufficient for dataset-building provenance. It must not require the UI database to reconstruct lineage later.

---

## 2. Required snippet artifacts

Each curated example must have a concrete snippet WAV artifact or a reproducible extraction contract.

### Preferred approach

Materialize snippet WAV files at curated-example export time.

### Why

- stable reusable source for repeated dataset builds
- avoids re-extracting from potentially moved source files
- simplifies dataset build ingestion

### If not materialized immediately

Then the manifest must contain a complete reproducible extraction spec, but this is less robust.

---

## Required CLI/API changes

## 1. Add curated export/materialization command

### New CLI command

`alpaca-pipelines prediction-review-materialize-curated`

### Inputs

- review session manifest or session export contract
- curated label assignments
- destination curated-source root
- optional overwrite/idempotency mode

### Outputs

JSON response with:
- curated source root
- manifest path
- item counts by label
- created/updated/skipped counts
- source collection/tape identifiers

### Purpose

Turn review outputs into reusable curated dataset sources.

---

## 2. Add curated source listing/status command

### New CLI command

`alpaca-pipelines curated-source-status`

### Outputs

- available curated source manifests
- counts by collection
- counts by label
- counts by provenance type
- stale/invalid source warnings if any

### Purpose

Support UI/catalog synchronization and sanity checking.

---

## 3. Extend dataset-build input contract

### Current limitation

Dataset build currently expects positive targets from hum-index entries only :contentReference[oaicite:7]{index=7}.

### Required change

Extend dataset-build config to support curated example sources as explicit build inputs.

### Suggested config additions

- `include_manual_review_curated: bool`
- `manual_review_curated_filters`
  - `collection_names`
  - `labels`
  - `prediction_run_ids`
  - `source_recording_keys`
- optional `manual_review_curated_weighting`
- optional `manual_review_curated_max_examples`

### Purpose

Allow dataset build to consume curated `target` and `noise` examples alongside existing indexed sources.

---

## 4. Extend dataset-status / manifest contracts

Dataset status and dataset manifest generation must surface curated provenance in build outputs.

### Required manifest fields per snippet

Each dataset snippet derived from curated review must include:

- `provenance_type: "manual_review_curated"`
- `curated_label`
- `source_collection_name`
- `source_category_dir`
- `source_relative_path`
- `source_recording_key` if available
- `source_prediction_run_id`
- `source_review_session_id`
- `source_review_item_id` or curated example id

### Required dataset meta summary

- total examples by provenance type
- total examples by label
- total examples by source collection
- total examples by source tape / source recording key
- total manually curated examples

---

## Dataset-builder changes

## 1. Add curated-source ingestion path

Extend dataset builder so it can read curated-source manifests in addition to existing indexed hum/clip/raw sources.

### Behavior

- `target` curated examples become valid positive examples
- `noise` curated examples become valid negative examples
- curated source items participate in split generation like other snippet sources

### Important

This removes the current hard dependency that positive examples must come only from hum index entries.

---

## 2. Keep source-type distinction in build logic

The builder must retain provenance identity while building datasets.

### Required provenance types at minimum

- `indexed_hum`
- `indexed_clip`
- `manual_review_curated`
- `raw_negative_source` or equivalent existing negative provenance

### Reason

Later inspection must distinguish manually curated examples from pre-existing labeled assets.

---

## 3. Preserve source recording lineage

Wherever dataset snippets are emitted, preserve lineage fields already aligned with existing provenance conventions such as `source_recording_key` :contentReference[oaicite:8]{index=8}.

### Required rule

A curated snippet must remain joinable to the original source recording/tape.

---

## 4. Support duplicate/overlap policy

Manual-review-curated items may overlap or duplicate existing hum-index entries or other curated items.

### Required initial policy

Implement explicit duplicate handling with one of:

- strict dedupe by source recording + start/end + label
- configurable overlap threshold dedupe
- no dedupe but explicit duplication markers

### Recommendation

Initial minimal implementation:
- exact-match dedupe on source recording identity + start/end + label
- include duplication summary in build report

---

## Manifest and contract changes

## 1. Curated-source manifest schema

Add a committed JSON schema for curated-source manifests.

### Suggested file

`contracts/json-schema/CuratedPredictionSourceManifest.json`

### Must define

- manifest-level source provenance
- per-item snippet provenance
- label taxonomy
- schema versioning

---

## 2. Extend dataset manifest schema

Add provenance fields for snippets and dataset summary.

### Required additions

#### `meta`
- `provenance_summary`
- `manual_curation_summary`

#### per snippet
- `provenance_type`
- `curated_label`
- `source_collection_name`
- `source_category_dir`
- `source_relative_path`
- `source_recording_key`
- `source_prediction_run_id`
- `source_review_session_id`
- `source_review_item_id`

### Compatibility

These should be additive fields, preserving backward compatibility for consumers that ignore them.

---

## 3. Extend run/output summaries where relevant

If curated materialization is modeled as a workflow operation or run-like job, emit JSON summaries containing:

- counts created
- counts updated
- counts by label
- source collection/tape summaries
- output manifest path(s)

---

## Pipeline API changes

If `PipelineAPI` exists for dataset and review workflows, add programmatic entry points:

- `materialize_curated_prediction_examples(...)`
- `list_curated_prediction_sources(...)`
- dataset-build support for curated source inclusion

These should mirror the CLI contracts.

---

## Error handling

Hard failures should occur for:

- curated export with zero eligible labeled reviewed items
- missing source recording references
- manifest/schema mismatch
- snippet materialization failure
- invalid label taxonomy
- dataset build configured to include curated examples but curated manifests are malformed

---

## Idempotency and mutability rules

## 1. Materialization must be idempotent

Re-running curated export for the same reviewed items should:
- update existing curated artifacts/manifest entries
- not create silent duplicates

## 2. Curated examples are versioned or replaceable

Choose one explicit policy:

### Option A — replace in place
- easiest initial implementation

### Option B — immutable versions with supersession
- better auditability

### Recommendation

Start with replace-in-place plus manifest timestamps and source IDs, unless stricter audit/version requirements already exist.

---

## Interaction with existing manual-review artifact commands

The current manual-review commands support preview/generate/export of review artifacts :contentReference[oaicite:9]{index=9}.

### Clarification

Those commands should remain focused on review-time artifacts.

### New requirement

Curated dataset-source materialization must be a separate contract, because:
- review artifacts are workflow artifacts
- curated examples are reusable dataset inputs

Do not overload Raven/manual-review artifact export as the long-term curated-source contract.

---

## Acceptance criteria

This work is complete in `alpaca-pipelines` when:

1. reviewed prediction segments with assigned labels can be materialized into a durable curated-source corpus
2. curated-source manifests include full provenance back to source tape, prediction run, and review item
3. dataset-build can consume curated `target` and `noise` examples as valid inputs
4. dataset build no longer requires all positive targets to originate only from hum index entries
5. dataset manifests expose provenance showing which examples came from `manual_review_curated`
6. dataset manifest/meta summaries support rollups by:
   - provenance type
   - label
   - collection
   - source tape / source recording key
7. repeated builds and repeated curated materialization are idempotent and auditable

---

## Recommended implementation order

1. define curated-source manifest schema
2. implement curated-example materialization command
3. materialize snippet WAVs + manifests
4. extend dataset-build config and ingestion
5. extend dataset manifest/meta provenance
6. expose status/listing commands
7. update UI integration after pipeline contracts are stable
