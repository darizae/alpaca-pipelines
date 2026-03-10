"""Evaluation pipeline package."""

from __future__ import annotations

from typing import Any

__all__ = ["EvaluationRunSpec", "execute_evaluation"]


def __getattr__(name: str) -> Any:
    if name == "EvaluationRunSpec":
        from alpaca_pipelines.evaluation.config import EvaluationRunSpec

        return EvaluationRunSpec
    if name == "execute_evaluation":
        from alpaca_pipelines.evaluation.executor import execute_evaluation

        return execute_evaluation
    raise AttributeError(name)
