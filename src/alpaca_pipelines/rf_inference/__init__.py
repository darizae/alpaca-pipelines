"""Standalone RF inference pipeline package."""

from __future__ import annotations

from typing import Any

__all__ = ["RfInferenceRunSpec", "execute_rf_inference"]


def __getattr__(name: str) -> Any:
    if name == "RfInferenceRunSpec":
        from alpaca_pipelines.rf_inference.config import RfInferenceRunSpec

        return RfInferenceRunSpec
    if name == "execute_rf_inference":
        from alpaca_pipelines.rf_inference.executor import execute_rf_inference

        return execute_rf_inference
    raise AttributeError(name)
