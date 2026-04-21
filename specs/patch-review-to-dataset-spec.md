# `alpaca-pipelines` — Follow-up Spec: Curated Contract Hardening and UI Handoff Compatibility

## Goal

Close the remaining gaps in `alpaca-pipelines` after the first curated-source implementation by ensuring:

1. the curated-source corpus has a committed JSON schema contract
2. source path provenance is normalized consistently
3. curated materialization can consume the UI’s durable curated export contract cleanly
4. dataset manifests preserve enough identifiers for exact UI-side provenance sync

---

## Problems to fix

### 1. Missing committed JSON schema for curated-source manifests

The code defines a Python model for curated manifests, but the curated-source format is not yet committed as a standalone JSON schema contract.

### 2. Inconsistent `source_relative_path` fallback behavior

When provenance fields are inferred from filesystem paths, the fallback currently risks storing a path shape inconsistent with the structured UI contract.

### 3. No first-class compatibility with UI durable curated inventory

Materialization still expects a review manifest and label file as separate inputs, rather than a stable exported curated-source contract from `alpaca-ui`.

### 4. Dataset manifests need exact curated IDs for downstream sync

The UI needs exact curated-example identifiers from built datasets to sync usage accurately.

---

## Scope

### In scope

* curated manifest schema contract
* provenance normalization rules
* curated materialization input contract extensions
* dataset manifest identifier guarantees

### Out of scope

* UI database changes
* frontend workflow
* non-curated dataset builder redesign

---

## Required changes

## 1. Commit JSON schema for curated-source manifests

### New required schema

Add:

`contracts/json-schema/CuratedPredictionSourceManifest.json`

### Must define

#### Top-level fields

* `schema_version`
* `source_type`
* `collection_name`
* `source_category_dir`
* `source_relative_path`
* `source_recording_key`
* `source_audio_file`
* `prediction_run_id`
* `review_session_id`
* `created_at`
* `items`

#### Per-item fields

* `curated_example_id`
* `review_item_id`
* `detection_index`
* `start_s`
* `end_s`
* `duration_s`
* `detection_score`
* `label`
* `snippet_wav_path`
* `source_recording_key`
* `source_collection_name`
* `source_category_dir`
* `source_relative_path`
* `source_audio_file`
* `prediction_run_id`
* `review_session_id`
* `provenance_type`
* `payload_json`

### Required tests

* schema is committed and loadable
* generated manifests validate against schema
* malformed manifests fail validation clearly

---

## 2. Normalize source provenance shape strictly

### New rule

For curated materialization and dataset manifests, source provenance must use the same structured shape as the UI contract:

* `source_collection_name`
* `source_category_dir`
* `source_relative_path`

### Required invariant

`source_relative_path` must be the file path **relative to the category directory**, not relative to collection root and not a display path.

### Add explicit derived field where helpful

If a combined path is needed, store it separately as:

* `source_display_path = "{collection}/{category}/{relative_path}"`

### Required fix

Update fallback inference logic so it does not collapse these concepts.

### Required validation

If inferred fields disagree with supplied fields, fail clearly instead of silently mixing shapes.

---

## 3. Add first-class UI-export input contract

### New purpose

`alpaca-pipelines` must accept a stable curated export produced by `alpaca-ui`, rather than only an ad hoc pair of:

* review manifest
* labels file

### New recommended contract

Add a second materialization input form:

`CuratedPredictionExportManifest`
with:

* review session identity
* source tape identity
* item-level review ids
* curated labels
* detection bounds
* source audio file
* provenance fields required for snippet materialization

### New CLI/API behavior

`prediction-review-materialize-curated` should accept either:

#### Mode A

existing:

* `--manifest`
* `--labels`

#### Mode B

new:

* `--curated-export-manifest`

### Programmatic API

Add support for:
`materialize_curated_prediction_examples(curated_export_manifest=...)`

### Compatibility rule

Both modes must produce the same curated corpus structure and same idempotency behavior.

---

## 4. Guarantee exact curated IDs survive into dataset manifests

### New rule

Every dataset snippet derived from curated review must preserve:

* `source_curated_example_id`

### Required manifest rule

This field must be present for all snippets whose:

* `provenance_type == "manual_review_curated"`

### Required dataset sync purpose

This enables `alpaca-ui` to map built dataset snippets back to durable curated examples exactly, without guessing by tape + bounds.

### Validation

Dataset build must fail if a curated-derived snippet is about to be emitted without:

* `source_curated_example_id`
* `source_review_session_id`
* `source_review_item_id`

---

## 5. Tighten dedupe and auditing guarantees

### Existing direction

The current implementation dedupes curated candidates by:

* source recording
* rounded start/end
* label

### Required audit additions

Include in materialization/build summaries:

* number of duplicates removed
* duplicate grouping key policy used
* whether any duplicates crossed source manifests

### Reason

This makes repeated materialization and repeated builds auditable, especially when UI-side curated inventory changes over time.

---

## Acceptance criteria

This follow-up is complete when:

1. curated-source manifests validate against a committed JSON schema
2. `source_relative_path` is normalized consistently with the UI contract
3. pipeline materialization can consume a first-class `alpaca-ui` curated export contract
4. all curated-derived dataset snippets preserve `source_curated_example_id`
5. dataset manifests support exact downstream provenance sync in `alpaca-ui`
6. dedupe behavior and duplicate counts are explicit in outputs and summaries
