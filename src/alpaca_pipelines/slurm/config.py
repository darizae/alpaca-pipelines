"""SLURM batch job configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SlurmConfig(BaseModel):
    """Configuration for SLURM batch script generation."""

    partition: str = "gpu"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 4
    gpus: int = 1
    gpu_type: str | None = None
    memory_gb: int = 32
    time_limit: str = "24:00:00"
    account: str | None = None
    job_name: str | None = None
    output_pattern: str = "slurm-%j.out"
    error_pattern: str = "slurm-%j.err"

    conda_env: str | None = None
    venv_path: str | None = None
    modules: list[str] = Field(default_factory=list)

    extra_sbatch_lines: list[str] = Field(default_factory=list)
    extra_env_vars: dict[str, str] = Field(default_factory=dict)

    def to_spec_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_spec_dict(cls, spec: dict[str, Any]) -> SlurmConfig:
        return cls.model_validate(spec)
