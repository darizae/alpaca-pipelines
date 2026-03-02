"""SLURM job generation and management."""

from alpaca_pipelines.slurm.config import SlurmConfig
from alpaca_pipelines.slurm.generator import generate_slurm_script

__all__ = ["SlurmConfig", "generate_slurm_script"]
