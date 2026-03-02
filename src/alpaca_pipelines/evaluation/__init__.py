"""Evaluation pipeline configuration and execution."""

from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.evaluation.executor import execute_evaluation

__all__ = ["EvaluationRunSpec", "execute_evaluation"]
