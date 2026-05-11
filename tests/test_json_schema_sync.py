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


def test_committed_json_schemas_match_current_models() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema_dir = repo_root / "contracts" / "json-schema"
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
        schema_path = schema_dir / f"{name}.json"
        with schema_path.open("r", encoding="utf-8") as handle:
            committed_schema = json.load(handle)
        assert committed_schema == model.model_json_schema()
