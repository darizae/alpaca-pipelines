from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


class RfClassifierProtocol(Protocol):
    """
    Minimal sklearn-like classifier protocol for RF inference.

    The mid-level orchestrator can pass in:
    - A scikit-learn RandomForestClassifier
    - Any compatible model object with predict_proba
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    @property
    def feature_names_in_(self) -> NDArray[np.str_] | None:  # sklearn exposes this on many estimators
        ...


FeatureValues = Mapping[str, float]
