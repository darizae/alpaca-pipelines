"""
Random Forest feature extraction utilities.

This module intentionally contains only low-level, orchestration-agnostic code:
- Feature extraction from in-memory audio arrays
- Schema alignment utilities
- A minimal protocol for sklearn-like RF classifiers

No file system conventions, no CLI, no selection table parsing.
"""

from bioacoustics_dl_toolbox.rf.features import (
    align_features_to_schema,
    compute_rf_features,
    mfcc_feature_schema,
    mfcc_summary,
    raven_robust_features,
)
from bioacoustics_dl_toolbox.rf.types import RfClassifierProtocol

__all__ = [
    "RfClassifierProtocol",
    "align_features_to_schema",
    "compute_rf_features",
    "mfcc_feature_schema",
    "mfcc_summary",
    "raven_robust_features",
]
