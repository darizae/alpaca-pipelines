## Spec — RF v1 end-to-end in `alpaca-pipelines`

Current touchpoints to inspect first: `src/alpaca_pipelines/rf_training/config.py`, `src/alpaca_pipelines/rf_training/executor.py`, `src/alpaca_pipelines/prediction/config.py`, `src/alpaca_pipelines/prediction/executor.py`, `src/alpaca_pipelines/rf/executor.py`

### 1. Goal

Adopt one shared RF feature contract for both RF training and RF inference. Remove the current split where training and inference rely on a toolbox-only `compute_rf_features(...)` path. RF training and RF inference must use the same in-repo feature code and the same persisted feature settings.

### 2. Shared RF feature module

Create a shared module under the pipeline repo, for example:

* `src/alpaca_pipelines/rf/audio_features/__init__.py`
* `src/alpaca_pipelines/rf/audio_features/robust_features.py`
* `src/alpaca_pipelines/rf/audio_features/mfcc_features.py`

Export exactly:

* `raven_robust_features`
* `mfcc_summary`

Implement the feature behavior exactly as this contract:

* `raven_robust_features(y, sr, t0, t1, fmin, fmax, n_fft=2048, hop_length=1024, window="hann", center=True)` returns:

  * `Dur 90% (s)`
  * `Dur 50% (s)`
  * `Center Freq (Hz)`
  * `Freq 5% (Hz)`
  * `Freq 25% (Hz)`
  * `Freq 75% (Hz)`
  * `Freq 95% (Hz)`
  * `BW 50% (Hz)`
  * `BW 90% (Hz)`
  * `Avg Entropy (bits)`
  * `Agg Entropy (bits)`

* `mfcc_summary(y, sr, t0, t1, n_mfcc=13, n_fft=2048, hop_length=1024, include_deltas=True)` returns:

  * `mfcc{i}_mean`, `mfcc{i}_std`
  * and, when enabled, `d_mfcc{i}_mean`, `d_mfcc{i}_std`, `dd_mfcc{i}_mean`, `dd_mfcc{i}_std`

**Critical:** RF training and RF inference must import only from this shared module. Do not keep two independent RF feature implementations.

### 3. RF training contract

Extend `RfTrainingRunSpec` with a nested RF feature config, for example:

* `feature_config.n_fft = 2048`
* `feature_config.hop_length = 1024`
* `feature_config.n_mfcc = 13`
* `feature_config.include_deltas = true`
* `feature_config.fmin_hz = 0`
* `feature_config.fmax_hz = 4000`

Keep existing sklearn RF hyperparameters unchanged:

* `random_state`
* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`
* `class_weight`
* `n_jobs`

Training behavior:

1. For each training and validation snippet, load mono audio.
2. Compute RF features over the full snippet window `[0.0, duration_s]`.
3. Use `fmin_hz` and `fmax_hz` from `feature_config`.
4. Build the training table from `raven_robust_features + mfcc_summary`.
5. Train `RandomForestClassifier` with the existing hyperparameters.
6. Persist:

   * model file
   * `rf_training_report.json`
   * model metadata that includes:

     * `feature_config`
     * `feature_names`
     * `feature_family = "rf_v1"`

The report must include `feature_config` in addition to the already persisted `feature_names` and sklearn hyperparameters. The current training executor already writes feature names and hyperparameters; extend that, do not redesign it

### 4. RF inference contract

Extend `PredictionRunSpec` with RF inference settings:

* `rf_threshold = 0.4`
* `rf_feature_config: null | same shape as training feature_config`

Keep existing:

* `apply_rf_filter`
* `rf_model_path`

Inference behavior:

1. When `apply_rf_filter=false`, behavior stays unchanged.
2. When `apply_rf_filter=true`, load the RF model and its persisted metadata.
3. If `rf_feature_config` is absent in the run spec, use the model’s persisted `feature_config`.
4. For each detection:

   * compute shared RF features on `[start_s, end_s]`
   * use `fmin_hz/fmax_hz` from the active RF feature config
5. Build the RF feature row from `raven_robust_features + mfcc_summary`.
6. Reorder columns to `feature_names_in_` when present.
7. Score with `predict_proba`.
8. Mark `rf_pass = rf_prob >= rf_threshold`.

Backward compatibility rule:

* If a detection payload already carries `cnn_logit_mean`, preserve it in the RF row only when the loaded model expects a `cnn_logit_mean` column.
* New RF training runs do not include `cnn_logit_mean` by default. Do not invent or synthesize this feature for new models.

### 5. Prediction artifact contract

The current prediction executor writes hashed per-file JSON names via `_prediction_output_path(...)`, while the RF executor separately derives `stem.json`; that must be unified

Implementation rule:

* RF inference must consume prediction outputs through the same path helper or through a persisted manifest.
* Do not derive RF input filenames independently from `Path(audio_file).stem`.

This is a release-blocking point.

### 6. Minimal pipeline tests

Add tests for:

1. shared feature parity: training extractor and inference extractor return identical columns for the same audio interval
2. metadata round-trip: RF training persists `feature_config` and inference consumes it
3. path contract: RF filtering reads the exact prediction files written by the prediction executor
4. schema export sync if JSON schemas are committed/generated
