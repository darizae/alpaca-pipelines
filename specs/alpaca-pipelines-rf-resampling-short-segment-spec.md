# Implementation Spec: RF Resampling and Short-Segment Standardization

Repository: `darizae/alpaca-pipelines`

## Purpose

Make Random Forest (RF) training and RF filtering compute features from a deterministic audio representation so short clips and short CNN detections do not produce non-finite MFCC delta features.

New RF feature extraction path:

```text
native audio
→ mono
→ resample to configured RF sample_rate_hz
→ slice requested segment
→ pad short segments to min_duration_s
→ compute Raven robust + MFCC features
```

No backward compatibility is required. Old RF model metadata and old RF run specs may be considered invalid.

## Required behavior

RF training and RF inference must use the same saved feature contract. The training run must persist the full RF feature config into `rf_model_metadata.json`, and prediction-time RF filtering must use that persisted config unless an explicit expert override is supplied.

Short segments must not create `NaN` delta-MFCC features. If `include_deltas=true`, short segments must be padded before MFCC delta computation.

## Files to change

### 1. Extend RF feature config

Path:

```text
src/alpaca_pipelines/rf/config.py
```

Replace `RfFeatureConfig` with a model containing these fields:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class RfFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate_hz: int = 48000
    min_duration_s: float = 0.4
    pad_short_segments: bool = True
    pad_mode: Literal["constant"] = "constant"

    n_fft: int = 2048
    hop_length: int = 1024
    n_mfcc: int = 13
    include_deltas: bool = True
    fmin_hz: float = 0.0
    fmax_hz: float = 4000.0
```

Validation:

```text
sample_rate_hz > 0
min_duration_s > 0
n_fft > 0
hop_length > 0
n_mfcc > 0
fmin_hz >= 0
fmax_hz >= fmin_hz
```

Use Pydantic validators. Do not allow extra keys.

### 2. Add shared RF audio preprocessing

Create:

```text
src/alpaca_pipelines/rf/audio_preprocess.py
```

Implement:

```python
from __future__ import annotations

import numpy as np
import librosa
from numpy.typing import NDArray

from alpaca_pipelines.rf.config import RfFeatureConfig


def to_mono(signal: NDArray[np.float32]) -> NDArray[np.float32]:
    ...


def resample_if_needed(
    signal: NDArray[np.float32],
    source_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    ...


def slice_seconds(
    signal: NDArray[np.float32],
    sr: int,
    t0: float,
    t1: float,
) -> NDArray[np.float32]:
    ...


def pad_to_min_duration(
    segment: NDArray[np.float32],
    sr: int,
    min_duration_s: float,
    pad_short_segments: bool,
) -> NDArray[np.float32]:
    ...


def prepare_rf_segment(
    signal: NDArray[np.float32],
    source_sr: int,
    t0: float,
    t1: float,
    config: RfFeatureConfig,
) -> tuple[NDArray[np.float32], int]:
    ...
```

Behavior:

```text
- Convert stereo/multichannel audio to mono by averaging channels.
- Resample with librosa.resample when source_sr != config.sample_rate_hz.
- Slice after resampling, using t0/t1 seconds.
- Clamp slice bounds to signal bounds.
- If the segment is empty and pad_short_segments=True, return a zero segment of min_duration_s.
- If the segment is shorter than min_duration_s and pad_short_segments=True, center-pad with zeros.
- If the segment is longer than min_duration_s, leave it unchanged.
- Return (segment.astype(np.float32), config.sample_rate_hz).
```

Do not crop long segments.

### 3. Refactor MFCC feature extraction

Path:

```text
src/alpaca_pipelines/rf/audio_features/mfcc_features.py
```

Change `mfcc_summary` so it expects an already-preprocessed segment. Remove `t0` and `t1` from the public signature.

New signature:

```python
def mfcc_summary(
    y: NDArray[np.float32],
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 1024,
    include_deltas: bool = True,
) -> dict[str, float]:
    ...
```

Required behavior:

```text
- If y is empty, raise ValueError. Empty/short handling belongs in audio_preprocess.py.
- Compute base MFCC summary.
- If include_deltas=True, compute first- and second-order deltas.
- If delta computation fails, raise ValueError with:
  - segment length in samples
  - sample rate
  - n_fft
  - hop_length
  - number of MFCC frames
- Do not silently fill delta features with NaN.
```

The helper that generates delta feature names may remain.

### 4. Refactor Raven robust feature extraction

Find the module exporting `raven_robust_features`, likely under:

```text
src/alpaca_pipelines/rf/audio_features/
```

Change it to expect an already-preprocessed segment. Remove internal `t0`/`t1` slicing.

New signature:

```python
def raven_robust_features(
    y: NDArray[np.float32],
    sr: int,
    fmin: float,
    fmax: float,
    n_fft: int,
    hop_length: int,
) -> dict[str, float]:
    ...
```

All RF segment slicing must happen in `prepare_rf_segment`.

### 5. Update RF training executor

Path:

```text
src/alpaca_pipelines/rf_training/executor.py
```

Update `_compute_features_for_file`.

New logic:

```python
signal, source_sr = _load_audio_signal(audio_path)
duration_s = float(len(signal)) / float(source_sr)

segment, rf_sr = prepare_rf_segment(
    signal=signal,
    source_sr=source_sr,
    t0=0.0,
    t1=duration_s,
    config=feature_config,
)

robust = raven_robust_features(
    y=segment,
    sr=rf_sr,
    fmin=feature_config.fmin_hz,
    fmax=feature_config.fmax_hz,
    n_fft=feature_config.n_fft,
    hop_length=feature_config.hop_length,
)

mfcc = mfcc_summary(
    y=segment,
    sr=rf_sr,
    n_mfcc=feature_config.n_mfcc,
    n_fft=feature_config.n_fft,
    hop_length=feature_config.hop_length,
    include_deltas=feature_config.include_deltas,
)
```

Keep the finite-feature validation. After this change, short padded segments should produce finite delta features.

The existing output writing should continue to persist:

```text
outputs/model/rf_model.joblib
outputs/model/rf_model_metadata.json
outputs/summaries/rf_training_report.json
```

The metadata/report `feature_config` must now include:

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

### 6. Update RF filter executor

Path:

```text
src/alpaca_pipelines/rf/executor.py
```

Inside `apply_rf_filter`, for each detection:

```python
segment, rf_sr = prepare_rf_segment(
    signal=signal,
    source_sr=file_sample_rate,
    t0=start_s,
    t1=end_s,
    config=active_feature_config,
)
```

Then pass `segment` and `rf_sr` into the refactored robust and MFCC feature functions.

Required behavior:

```text
- Use metadata-derived feature_config by default.
- Use explicit rf_feature_config only when supplied.
- Apply the same resampling and padding policy used during RF training.
- Continue writing *_rf_filtered.json.
- Continue writing rf_score, rf_pass, rf_filtered, rf_model_path, rf_threshold.
```

### 7. Update prediction config schema

Inspect and update:

```text
src/alpaca_pipelines/prediction/config.py
contracts/json-schema/*
```

Ensure `rf_feature_config` accepts the new `RfFeatureConfig` fields. The prediction executor should continue passing:

```python
spec.rf_feature_config.model_dump() if spec.rf_feature_config is not None else None
```

into `apply_rf_filter`.

### 8. Update tests

Path:

```text
tests/test_rf_alignment_contract.py
```

Remove or replace the test that expects short MFCC intervals to fill delta features with `NaN`.

Add tests:

```text
test_prepare_rf_segment_resamples_to_configured_sample_rate
test_prepare_rf_segment_pads_short_segment_to_min_duration
test_mfcc_summary_short_padded_segment_produces_finite_delta_features
test_rf_training_short_clip_with_deltas_completes_without_nan
test_rf_training_persists_new_feature_config_metadata
test_rf_filter_uses_persisted_resample_and_padding_config
test_training_and_inference_feature_columns_still_match
```

Assertions:

```text
- A 20 ms input segment with sample_rate_hz=48000 and min_duration_s=0.4 becomes exactly 19200 samples.
- Short padded segments produce finite d_mfcc* and dd_mfcc* values.
- RF training with include_deltas=true completes on a dataset containing a very short snippet.
- rf_model_metadata.json includes the full new feature_config.
- RF filtering a short detection writes rf_score and boolean rf_pass.
- Training and inference feature column order still matches.
```

### 9. Acceptance criteria

RF training:

```text
Given a dataset containing a 20 ms WAV snippet,
when RF Training runs with include_deltas=true, sample_rate_hz=48000, min_duration_s=0.4, pad_short_segments=true,
then feature extraction completes,
and all d_mfcc* and dd_mfcc* values are finite,
and rf_model.joblib, rf_model_metadata.json, and rf_training_report.json are written.
```

RF filtering:

```text
Given a prediction JSON containing a detection shorter than 400 ms,
when apply_rf_filter runs with metadata-derived feature_config,
then the detection segment is resampled and padded,
and *_rf_filtered.json contains rf_score and boolean rf_pass.
```

Metadata:

```text
rf_model_metadata.json must contain the full feature_config and feature_family="rf_v1".
```

### 10. Implementation order

```text
1. Extend RfFeatureConfig.
2. Add audio_preprocess.py.
3. Refactor MFCC and robust feature functions.
4. Update RF training executor.
5. Update RF filter executor.
6. Update prediction config/schema.
7. Update tests.
8. Run test suite.
```
