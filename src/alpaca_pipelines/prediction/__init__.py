"""Prediction pipeline configuration and execution."""

from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.prediction.executor import execute_prediction

__all__ = ["PredictionRunSpec", "execute_prediction"]
