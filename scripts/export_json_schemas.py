from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from alpaca_pipelines.contracts import PredictionReviewIndexSummary, RunState
from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.prediction.review import (
    CuratedPredictionExportManifest,
    CuratedPredictionSourceManifest,
    PredictionReviewSessionManifest,
    PredictionReviewSpectrogramConfig,
)
from alpaca_pipelines.rf_training.config import RfTrainingRunSpec
from alpaca_pipelines.slurm.config import SlurmConfig
from alpaca_pipelines.training.config import TrainingRunSpec


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "contracts" / "json-schema"
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, type[BaseModel]] = {
        "TrainingRunSpec": TrainingRunSpec,
        "PredictionRunSpec": PredictionRunSpec,
        "PredictionReviewSpectrogramConfig": PredictionReviewSpectrogramConfig,
        "PredictionReviewSessionManifest": PredictionReviewSessionManifest,
        "CuratedPredictionExportManifest": CuratedPredictionExportManifest,
        "CuratedPredictionSourceManifest": CuratedPredictionSourceManifest,
        "EvaluationRunSpec": EvaluationRunSpec,
        "RfTrainingRunSpec": RfTrainingRunSpec,
        "SlurmConfig": SlurmConfig,
        "RunState": RunState,
        "PredictionReviewIndexSummary": PredictionReviewIndexSummary,
    }
    for name, model in models.items():
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(model.model_json_schema(), handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
