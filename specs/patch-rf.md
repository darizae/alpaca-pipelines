## Patch spec — RF v1 cleanup, no legacy `cnn_logit_mean`

This patch applies on top of PR #2 (`Align RF`) and resolves the remaining blockers while making `cnn_logit_mean` unsupported and fully removed from the RF v1 path. Current PR already adds shared RF feature code, `RfFeatureConfig`, persisted RF metadata, and exact hashed prediction-path handoff.

### 1. Prevent short-window MFCC delta crashes

Files:

* `src/alpaca_pipelines/rf/audio_features/mfcc_features.py`
* `tests/test_rf_alignment_contract.py`

Required change:

* Make `mfcc_summary()` robust for very short intervals.
* Keep the same output schema and column names.
* When `include_deltas=True` but the MFCC frame count is too short for `librosa.feature.delta(...)`:

  * compute base `mfcc{i}_mean` / `mfcc{i}_std` normally
  * emit all `d_mfcc{i}_*` and `dd_mfcc{i}_*` fields as `NaN`
  * do not raise

Implementation requirement:

* Handle this inside `mfcc_summary()`, either by:

  * pre-checking frame count before delta computation, or
  * catching the librosa error and filling delta fields with `NaN`

Required test:

* Add a test using a very short interval that asserts:

  * no exception
  * all expected MFCC / delta / delta-delta columns are present
  * delta columns are `NaN` for the too-short case

### 2. Remove `cnn_logit_mean` from RF v1 entirely

Files:

* `src/alpaca_pipelines/rf/executor.py`
* `specs/align-rf.md`
* `tests/test_rf_alignment_contract.py`

Required behavior change:

* RF v1 does not support `cnn_logit_mean`.
* Remove the legacy compatibility branch in `apply_rf_filter()` that conditionally reads `cnn_logit_mean` from detections.
* RF inference feature rows must be built from exactly:

  * `raven_robust_features(...)`
  * `mfcc_summary(...)`

Hard failure rule:

* If loaded model metadata or model feature names indicate a required `cnn_logit_mean` column, fail immediately with a clear error such as:

  * `Unsupported RF model: legacy feature 'cnn_logit_mean' is not supported by rf_v1`

Do not:

* compute `cnn_logit_mean`
* preserve `cnn_logit_mean`
* synthesize a replacement from detection `score`
* document any backward compatibility path for it

Spec change:

* Update `specs/align-rf.md` section 4 to remove the current backward compatibility rule and replace it with:

  * RF v1 does not support `cnn_logit_mean`
  * models requiring `cnn_logit_mean` are invalid for RF v1 inference

Required test:

* Add a test that loads metadata or model feature names containing `cnn_logit_mean` and asserts RF inference fails before scoring

### 3. Fix dataset-mode validation for `apply_rf_filter`

Files:

* `src/alpaca_pipelines/prediction/config.py`
* `contracts/json-schema/PredictionRunSpec.json` if committed/generated

Required change:

* `PredictionRunSpec` must require `rf_model_path` whenever `apply_rf_filter=True`, for all modes:

  * `tape`
  * `dataset`
  * `collection`

Implementation requirement:

* Move the `apply_rf_filter -> rf_model_path required` validation out of the tape/collection-only branches
* Apply it once across all modes

Required test:

* Add a validation test for:

  * `mode="dataset"`
  * `apply_rf_filter=True`
  * missing `rf_model_path`
* Expect validation failure

### 4. Add hard `feature_family == "rf_v1"` validation on load

Files:

* `src/alpaca_pipelines/rf/executor.py`
* `tests/test_rf_alignment_contract.py`

Required change:

* RF inference must require valid model metadata.
* After loading `rf_model_metadata.json`, enforce:

  * metadata file exists
  * metadata is an object
  * `feature_family` exists
  * `feature_family == "rf_v1"`

Failure behavior:

* Raise a hard error before feature extraction or scoring if:

  * metadata is missing
  * `feature_family` is missing
  * `feature_family` is anything other than `"rf_v1"`

Required test:

* Add a test with wrong `feature_family`
* Assert inference fails before scoring

### 5. Add one real end-to-end metadata round-trip test

Files:

* `tests/test_rf_alignment_contract.py`

Required test flow:

1. Run `execute_rf_training()` with a non-default `RfFeatureConfig`
2. Use the produced:

   * `rf_model.joblib`
   * `rf_model_metadata.json`
3. Run `apply_rf_filter()` with:

   * `rf_model_path` pointing to the trained artifact
   * `rf_feature_config=None`
4. Assert:

   * inference succeeds
   * metadata-sourced `feature_config` is accepted and used
   * RF scores are produced

Test requirement:

* This must use real training output from `execute_rf_training()`, not hand-written metadata

### 6. Keep these existing PR behaviors unchanged

Do not regress:

* exact prediction artifact handoff through `prediction_inputs`, not `Path(audio_file).stem` reconstruction
* persisted RF metadata written by training:

  * `feature_family`
  * `feature_names`
  * `feature_config`

## Done criteria

This PR is unblocked only when all are true:

* short detection windows cannot crash RF inference through MFCC delta computation
* `cnn_logit_mean` is fully unsupported and explicitly rejected for RF v1
* dataset mode rejects `apply_rf_filter=true` without `rf_model_path`
* RF inference hard-rejects missing or non-`rf_v1` metadata
* one real training→persist→inference metadata round-trip test passes
