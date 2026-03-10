"""Prediction pipeline package."""

from __future__ import annotations

from typing import Any

__all__ = ["PredictionRunSpec", "execute_prediction"]


def __getattr__(name: str) -> Any:
    if name == "PredictionRunSpec":
        from alpaca_pipelines.prediction.config import PredictionRunSpec

        return PredictionRunSpec
    if name == "execute_prediction":
        from alpaca_pipelines.prediction.executor import execute_prediction

        return execute_prediction
    raise AttributeError(name)
