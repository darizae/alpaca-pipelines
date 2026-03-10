"""RF training pipeline package."""

from __future__ import annotations

from typing import Any

__all__ = ["RfTrainingRunSpec", "execute_rf_training"]


def __getattr__(name: str) -> Any:
    if name == "RfTrainingRunSpec":
        from alpaca_pipelines.rf_training.config import RfTrainingRunSpec

        return RfTrainingRunSpec
    if name == "execute_rf_training":
        from alpaca_pipelines.rf_training.executor import execute_rf_training

        return execute_rf_training
    raise AttributeError(name)
