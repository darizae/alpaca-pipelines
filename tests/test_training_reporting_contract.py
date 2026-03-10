from __future__ import annotations

import logging
from pathlib import Path

from alpaca_pipelines.runs.manager import RunManager
from alpaca_pipelines.training.config import TrainingRunSpec
from alpaca_pipelines.training.executor import (
    _best_metric_from_history,
    _build_training_summary_payload,
    _TrainingMetricsCollector,
)


def test_run_manager_create_run_only_populates_type_specific_output_pointers(
    tmp_path: Path,
) -> None:
    manager = RunManager(tmp_path / "runs")

    training_run = manager.create_run(
        "training", TrainingRunSpec(dataset_name="dataset-a").to_spec_dict()
    )
    prediction_run = manager.create_run("prediction", {"model_path": "/models/final.pt"})

    assert training_run.outputs.model_dir is not None
    assert training_run.outputs.summaries_dir is not None
    assert training_run.outputs.predictions_dir is None
    assert training_run.outputs.evaluation_dir is None
    assert prediction_run.outputs.predictions_dir is not None
    assert prediction_run.outputs.prediction_selection_tables_summary_path is not None
    assert prediction_run.outputs.model_dir is None
    assert prediction_run.outputs.summaries_dir is None


def test_training_metrics_collector_builds_history_and_summary_payload() -> None:
    collector = _TrainingMetricsCollector()
    for message in [
        "train|0|loss:0.628|accuracy:0.671|f1:0.048|precision:0.464|recall:0.025|lr:1.00e-05|t:6.3",
        "val|0|loss:0.625|accuracy:0.654|f1:0.560|precision:0.497|recall:0.643|t:0.8",
        "train|2|loss:0.397|accuracy:0.819|f1:0.671|precision:0.833|recall:0.562|lr:1.00e-05|t:4.2",
        "val|2|loss:0.264|accuracy:0.896|f1:0.823|precision:0.994|recall:0.702|t:0.5",
        "test|0|loss:0.096|accuracy:0.971|f1:0.959|precision:0.955|recall:0.963|t:0.9",
    ]:
        collector.emit(logging.LogRecord("x", logging.INFO, __file__, 1, message, (), None))

    history = collector.build_history_payload()
    best_value, best_epoch = _best_metric_from_history(history, "accuracy", "max")
    payload = _build_training_summary_payload(
        spec=TrainingRunSpec(dataset_name="dataset-a", positive_class="target"),
        total_epochs=40,
        history=history,
        best_metric_name="accuracy",
        best_metric_value=best_value,
        best_epoch=best_epoch,
        model_output_path=Path("/runs/training/run-a/outputs/model/trained_model.pt"),
        tensorboard_dir="/runs/training/run-a/outputs/summaries/run",
    )

    assert history["train"][-1]["epoch"] == 2
    assert best_value == 0.896
    assert best_epoch == 2
    assert payload["current_epoch"] == 3
    assert payload["test_metrics"]["accuracy"] == 0.971
    assert payload["tensorboard_dir"] == "/runs/training/run-a/outputs/summaries/run"
