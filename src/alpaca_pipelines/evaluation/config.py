"""
Evaluation run specification.

Evaluates predictions against ground truth from a dataset,
computing classification metrics and generating reports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluationRunSpec(BaseModel):
    """Complete specification for an evaluation run."""

    prediction_run_id: str | None = None
    predictions_dir: str | None = None

    dataset_name: str

    detection_thresholds: list[float] = Field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    split: Literal["train", "val", "test"] = "test"

    sequence_length: int | None = None
    batch_size: int = 16
    num_workers: int = 4
    use_cuda: bool = True

    run_name: str = ""

    def to_spec_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_spec_dict(cls, spec: dict[str, Any]) -> EvaluationRunSpec:
        return cls.model_validate(spec)
