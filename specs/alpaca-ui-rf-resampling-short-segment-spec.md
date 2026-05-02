# Implementation Spec: UI Support for RF Resampling and Short-Segment Standardization

Repository: `darizae/alpaca-ui`

## Purpose

Expose the new RF feature contract added in `darizae/alpaca-pipelines`:

```text
sample_rate_hz
min_duration_s
pad_short_segments
pad_mode
n_fft
hop_length
n_mfcc
include_deltas
fmin_hz
fmax_hz
```

The UI must let users configure these fields during RF Training and must show/pass the full feature config during Prediction RF expert override. Default prediction behavior should continue to inherit the RF feature config from the selected RF training run metadata.

No backward compatibility is required.

## Required behavior

RF Training page:

```text
- Show/edit sample_rate_hz.
- Show/edit min_duration_s.
- Show/edit pad_short_segments.
- Show pad_mode as "constant".
- Keep existing FFT/hop/MFCC/frequency/delta fields.
- Submit the complete feature_config in RF training payloads.
```

Prediction page:

```text
- When RF filter is enabled and an RF training run is selected, display the full resolved RF feature config.
- With expert override disabled, send rf_feature_config: null so the pipeline inherits metadata.
- With expert override enabled, send the full RFFeatureConfig including new fields.
```

## Files to change

### 1. Update frontend type definitions

Path:

```text
packages/frontend/src/types/index.ts
```

Replace `RFFeatureConfig` with:

```ts
export interface RFFeatureConfig {
  sample_rate_hz: number;
  min_duration_s: number;
  pad_short_segments: boolean;
  pad_mode: "constant";

  n_fft: number;
  hop_length: number;
  n_mfcc: number;
  include_deltas: boolean;
  fmin_hz: number;
  fmax_hz: number;
}
```

This type is used by:

```text
RFTrainingConfig
PredictionConfig
RFTrainingRunSummary
Prediction RF expert override state
```

### 2. Update frontend defaults

Path:

```text
packages/frontend/src/utils/defaults.ts
```

Replace `DEFAULT_RF_FEATURE_CONFIG` with:

```ts
export const DEFAULT_RF_FEATURE_CONFIG: RFFeatureConfig = {
  sample_rate_hz: 48000,
  min_duration_s: 0.4,
  pad_short_segments: true,
  pad_mode: "constant",

  n_fft: 2048,
  hop_length: 1024,
  n_mfcc: 13,
  include_deltas: true,
  fmin_hz: 0,
  fmax_hz: 4000,
};
```

Rationale:

```text
sample_rate_hz=48000 aligns with the UI CNN sample_rate default.
min_duration_s=0.4 aligns with the UI CNN sequence_length_ms default of 400 ms.
n_fft/hop/frequency defaults keep the existing RF/CNN UI alignment.
```

### 3. Update RF Training page

Path:

```text
packages/frontend/src/pages/NewRFTrainingPage.tsx
```

In the existing **RF Feature Settings** accordion, add controls for:

```text
sample_rate_hz
min_duration_s
pad_short_segments
pad_mode
```

Recommended UI fields:

```tsx
<NumericTextField
  label="sample_rate_hz"
  value={config.feature_config.sample_rate_hz}
  onChange={(e) => updateFeatureConfig({ sample_rate_hz: Number(e.target.value) })}
  slotProps={{ htmlInput: { min: 1, step: 1 } }}
/>

<NumericTextField
  label="min_duration_s"
  value={config.feature_config.min_duration_s}
  onChange={(e) => updateFeatureConfig({ min_duration_s: Number(e.target.value) })}
  slotProps={{ htmlInput: { min: 0.001, step: 0.01 } }}
/>

<FormControlLabel
  control={
    <Switch
      checked={config.feature_config.pad_short_segments}
      onChange={(e) => updateFeatureConfig({ pad_short_segments: e.target.checked })}
    />
  }
  label="Pad short segments"
/>

<TextField
  label="pad_mode"
  value={config.feature_config.pad_mode}
  slotProps={{ input: { readOnly: true } }}
/>
```

Validation additions in `validateConfig()`:

```text
sample_rate_hz >= 1
min_duration_s > 0
pad_mode === "constant"
```

Keep existing validation:

```text
n_fft >= 1
hop_length >= 1
n_mfcc >= 1
fmin_hz >= 0
fmax_hz >= fmin_hz
```

The RF Training payload must include the full `feature_config`.

### 4. Update Prediction page RF advanced config

Path:

```text
packages/frontend/src/pages/NewPredictionPage.tsx
```

Update the RF Filter Advanced section so it displays and optionally overrides all fields in `RFFeatureConfig`.

Required behavior:

```text
- `resolvedRfFeatureConfig` must include the new fields from the selected RF training run.
- `effectiveRfFeatureConfig` must include the new fields.
- When expert override is disabled, normalizePayload() must send rf_feature_config: null.
- When expert override is enabled, normalizePayload() must send the full new config.
```

Add validation in `validateConfig()` when `effectiveRfFeatureConfig` is present:

```text
sample_rate_hz >= 1
min_duration_s > 0
pad_mode === "constant"
n_fft >= 1
hop_length >= 1
n_mfcc >= 1
fmin_hz >= 0
fmax_hz >= fmin_hz
```

Do not partially fill missing new fields. No backward compatibility is required.

### 5. Update upstream RF defaults extraction

Path:

```text
packages/frontend/src/utils/upstreamRunDefaults.ts
```

Update `getPredictionRFDefaultsFromRFTrainingRun`.

Behavior:

```text
- Read full feature_config from selected RF training run summary/details or spec.
- Return a complete RFFeatureConfig with the new fields.
- Do not silently synthesize missing new fields from old runs.
```

Preferred source order:

```text
1. run.reporting/summary details feature_config if available
2. run.spec.feature_config
```

If the selected completed RF training run lacks the new fields, return `rf_feature_config: null` and let the UI show an error that the run is not compatible with the current RF feature contract.

### 6. Update backend tests

Path:

```text
packages/backend/tests/test_pipeline_service.py
```

Update any expected `rf_feature_config` object from the old shape:

```json
{
  "n_fft": 2048,
  "hop_length": 1024,
  "n_mfcc": 13,
  "include_deltas": true,
  "fmin_hz": 0,
  "fmax_hz": 4000
}
```

to the new shape:

```json
{
  "sample_rate_hz": 48000,
  "min_duration_s": 0.4,
  "pad_short_segments": true,
  "pad_mode": "constant",
  "n_fft": 2048,
  "hop_length": 1024,
  "n_mfcc": 13,
  "include_deltas": true,
  "fmin_hz": 0,
  "fmax_hz": 4000
}
```

Relevant tests include prediction creation with RF filter and explicit `rf_feature_config`.

No backend service behavior change should be necessary if it already forwards arbitrary JSON config to the pipeline CLI. Only expectations need to reflect the new full config.

### 7. Update frontend docs

Path:

```text
packages/frontend/src/content/user-wiki.md
```

Update **RF Training Runs**:

```text
RF Feature Settings include sample_rate_hz and min_duration_s. RF training resamples snippets to sample_rate_hz and pads snippets shorter than min_duration_s before computing MFCC delta features. Defaults are 48 kHz and 0.4 s, matching the UI CNN defaults.
```

Update **Prediction Runs**:

```text
RF filtering inherits the RF model's saved feature config, including sample rate and short-segment padding, unless expert override is enabled.
```

### 8. Acceptance criteria

RF Training UI:

```text
- A new RF training run payload includes sample_rate_hz, min_duration_s, pad_short_segments, and pad_mode.
- Default values are 48000, 0.4, true, and "constant".
- Invalid values are blocked before submission.
```

Prediction UI:

```text
- Selecting a completed compatible RF training run shows the full RF feature config.
- Expert override disabled sends rf_feature_config: null.
- Expert override enabled sends the full new RFFeatureConfig.
- Incompatible old RF training runs without the new fields are not silently accepted.
```

Backend tests:

```text
- RF prediction payload tests assert the new feature_config shape.
- Existing inheritance behavior with rf_feature_config: null still passes.
```

Docs:

```text
- User guide explains RF resampling and short-segment padding.
```

### 9. Implementation order

```text
1. Update TypeScript RFFeatureConfig type.
2. Update DEFAULT_RF_FEATURE_CONFIG.
3. Update NewRFTrainingPage form and validation.
4. Update upstreamRunDefaults.ts.
5. Update NewPredictionPage RF advanced display/override/validation.
6. Update backend tests.
7. Update user-wiki.md.
8. Run frontend typecheck and backend tests.
```
