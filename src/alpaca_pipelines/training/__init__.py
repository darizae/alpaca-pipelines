"""Training pipeline package."""

from __future__ import annotations

from typing import Any

__all__ = ["TrainingRunSpec", "execute_training"]


def __getattr__(name: str) -> Any:
    if name == "TrainingRunSpec":
        from alpaca_pipelines.training.config import TrainingRunSpec

        return TrainingRunSpec
    if name == "execute_training":
        from alpaca_pipelines.training.executor import execute_training

        return execute_training
    raise AttributeError(name)
