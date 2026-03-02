"""Training pipeline configuration and execution."""

from alpaca_pipelines.training.config import TrainingRunSpec
from alpaca_pipelines.training.executor import execute_training

__all__ = ["TrainingRunSpec", "execute_training"]
