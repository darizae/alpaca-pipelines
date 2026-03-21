"""
Evaluation pipeline executor.

Computes classification metrics on the test split of a dataset
given a trained model or existing predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.dataset.loader import DatasetHandle, load_dataset_handle
from alpaca_pipelines.evaluation.config import EvaluationRunSpec
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.runs.manager import RunManager
from bioacoustics_dl_toolbox.config import (
    AugmentationConfig,
    ClassifierConfig,
    EncoderConfig,
    NormalizationConfig,
    SpectrogramConfig,
)
from bioacoustics_dl_toolbox.logging.logger import create_logger
from bioacoustics_dl_toolbox.metrics.auc import AUCMeter
from bioacoustics_dl_toolbox.metrics.confusion import ConfusionMeter
from bioacoustics_dl_toolbox.metrics.core import (
    FPR,
    Accuracy,
    F1Score,
    Precision,
    Recall,
)

logger = logging.getLogger(__name__)


def _resolve_target_index(class_to_index: dict[str, int]) -> int:
    """Determine the target class index from the saved class mapping.

    Hard-fails if "target" is not present in the mapping.
    """
    if "target" not in class_to_index:
        raise ValueError(
            "Model class mapping does not contain 'target'. Available classes: {}".format(
                sorted(class_to_index.keys())
            )
        )
    return class_to_index["target"]


def _validate_class_to_index(class_to_index: dict[str, int]) -> None:
    """Validate that class_to_index values form a contiguous range [0..N-1]."""
    indices = sorted(class_to_index.values())
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            "Model class_to_index values are not contiguous [0..{}]: got {}".format(
                len(indices) - 1, indices
            )
        )


def _validate_dataset_matches_model(
    dataset_class_to_index: dict[str, int],
    model_class_to_index: dict[str, int],
) -> None:
    if dataset_class_to_index != model_class_to_index:
        raise ValueError(
            "Dataset class_to_index does not match model class_to_index. "
            "Dataset mapping: {}. Model mapping: {}.".format(
                dataset_class_to_index, model_class_to_index
            )
        )


def _evaluate_dataset_split(
    model: nn.Module,
    dataset_handle: DatasetHandle,
    split_name: str,
    spec_config: SpectrogramConfig,
    norm_config: NormalizationConfig,
    sequence_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    thresholds: list[float],
    target_index: int,
) -> dict[str, Any]:
    """Run model inference on a dataset split and compute metrics."""
    from bioacoustics_dl_toolbox.audio.datasets import SpectrogramDataset

    split_files = dataset_handle.splits.all_splits[split_name]
    aug_config = AugmentationConfig(enabled=False)

    dataset = SpectrogramDataset(
        file_names=split_files,
        working_dir=str(dataset_handle.snippets_dir),
        spec_config=spec_config,
        norm_config=norm_config,
        aug_config=aug_config,
        classes=dataset_handle.classes,
        sequence_length=sequence_length,
        class_to_index=dataset_handle.class_to_index,
        dataset_name=split_name,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_labels: list[int] = []
    all_scores: list[float] = []
    all_predictions: list[int] = []

    model.eval()
    with torch.no_grad():
        for features, label_dict in loader:
            features = features.to(device)
            call_labels = label_dict["call"].to(device, dtype=torch.int64)

            output = model(features)
            probabilities = nn.functional.softmax(output, dim=1)

            predicted_classes = torch.argmax(output, dim=1)
            target_scores = probabilities[:, target_index]

            all_labels.extend(call_labels.cpu().numpy().tolist())
            all_scores.extend(target_scores.cpu().numpy().tolist())
            all_predictions.extend(predicted_classes.cpu().numpy().tolist())

    labels_tensor = torch.tensor(all_labels)
    predictions_tensor = torch.tensor(all_predictions)
    scores_array = np.array(all_scores)
    labels_array = np.array(all_labels)

    accuracy_metric = Accuracy()
    f1_metric = F1Score()
    precision_metric = Precision()
    recall_metric = Recall()
    fpr_metric = FPR()

    accuracy_metric.update(labels_tensor, predictions_tensor)
    f1_metric.update(labels_tensor, predictions_tensor)
    precision_metric.update(labels_tensor, predictions_tensor)
    recall_metric.update(labels_tensor, predictions_tensor)
    fpr_metric.update(labels_tensor, predictions_tensor)

    auc_meter = AUCMeter()
    auc_meter.add(scores_array, labels_array)
    auc_value, tpr_curve, fpr_curve = auc_meter.value()

    num_classes = dataset_handle.num_classes
    confusion_meter = ConfusionMeter(num_classes)
    confusion_meter.add(
        np.array(all_predictions),
        np.array(all_labels),
    )
    confusion_raw = confusion_meter.confusion.numpy().tolist()
    confusion_normalized = confusion_meter.value().numpy().tolist()

    threshold_metrics: list[dict[str, Any]] = []
    for threshold in thresholds:
        thresholded_predictions = torch.tensor(
            [1 if score >= threshold else 0 for score in all_scores]
        )
        threshold_accuracy = Accuracy()
        threshold_f1 = F1Score()
        threshold_precision = Precision()
        threshold_recall = Recall()

        threshold_accuracy.update(labels_tensor, thresholded_predictions)
        threshold_f1.update(labels_tensor, thresholded_predictions)
        threshold_precision.update(labels_tensor, thresholded_predictions)
        threshold_recall.update(labels_tensor, thresholded_predictions)

        threshold_metrics.append(
            {
                "threshold": threshold,
                "accuracy": round(threshold_accuracy.get(), 6),
                "f1": round(threshold_f1.get(), 6),
                "precision": round(threshold_precision.get(), 6),
                "recall": round(threshold_recall.get(), 6),
            }
        )

    return {
        "split": split_name,
        "n_samples": len(all_labels),
        "n_positive": sum(all_labels),
        "n_negative": len(all_labels) - sum(all_labels),
        "metrics": {
            "accuracy": round(accuracy_metric.get(), 6),
            "f1": round(f1_metric.get(), 6),
            "precision": round(precision_metric.get(), 6),
            "recall": round(recall_metric.get(), 6),
            "fpr": round(fpr_metric.get(), 6),
            "auc": round(float(auc_value), 6),
        },
        "threshold_sweep": threshold_metrics,
        "confusion_matrix_raw": confusion_raw,
        "confusion_matrix_normalized": confusion_normalized,
        "class_names": dataset_handle.classes,
        "roc_curve": {
            "tpr": tpr_curve.tolist(),
            "fpr": fpr_curve.tolist(),
        },
    }


def execute_evaluation(
    run_state: RunState,
    environment: PipelineEnvironment,
    run_manager: RunManager,
) -> RunState:
    """Execute an evaluation run from its persisted specification."""
    spec = EvaluationRunSpec.from_spec_dict(run_state.spec)
    run_dir = Path(run_state.run_dir)

    run_state = run_manager.mark_running(run_state.run_id)

    try:
        run_name = spec.run_name or run_state.run_id
        evaluation_logger = create_logger(
            run_name,
            debug=False,
            log_dir=str(run_dir / "logs"),
        )

        dataset_dir = environment.resolve_dataset_dir(spec.dataset_name)
        dataset_handle = load_dataset_handle(dataset_dir, environment.collection_root)

        if dataset_handle.num_classes != 2:
            raise ValueError(
                "Evaluation threshold sweep requires a binary dataset (2 classes). "
                "Got {} classes: {}".format(dataset_handle.num_classes, dataset_handle.classes)
            )

        model_path: str | None = None
        if spec.prediction_run_id is not None:
            prediction_state = run_manager.find_run(spec.prediction_run_id)
            if prediction_state.outputs.trained_model_path is not None:
                model_path = prediction_state.outputs.trained_model_path
            prediction_spec = prediction_state.spec
            if model_path is None and "model_path" in prediction_spec:
                model_path = prediction_spec["model_path"]
        elif spec.predictions_dir is not None:
            summary_path = Path(spec.predictions_dir) / "prediction_summary.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                model_path = summary.get("model_path")

        if model_path is None:
            raise ValueError(
                "Cannot determine model path for evaluation. "
                "Provide prediction_run_id or predictions_dir with a prediction_summary.json."
            )

        evaluation_logger.info("Loading model from: {}".format(model_path))
        from bioacoustics_dl_toolbox.models.classifier import Classifier
        from bioacoustics_dl_toolbox.models.encoder import ResidualEncoder
        from bioacoustics_dl_toolbox.training.checkpoints import load_model

        model_dict = load_model(model_path)
        encoder_config = EncoderConfig(**model_dict["encoderConfig"])
        classifier_config = ClassifierConfig(**model_dict["classifierConfig"])
        spec_config = SpectrogramConfig(**model_dict["spectrogramConfig"])

        class_to_index: dict[str, int] = model_dict["classes"]
        _validate_class_to_index(class_to_index)
        _validate_dataset_matches_model(dataset_handle.class_to_index, class_to_index)
        target_index = _resolve_target_index(class_to_index)

        encoder = ResidualEncoder(encoder_config)
        classifier = Classifier(classifier_config)
        encoder.load_state_dict(model_dict["encoderState"])
        classifier.load_state_dict(model_dict["classifierState"])
        model = nn.Sequential(encoder, classifier)

        device = torch.device("cuda" if spec.use_cuda and torch.cuda.is_available() else "cpu")
        model = model.to(device)
        evaluation_logger.info("Device: {}".format(device))

        norm_config = NormalizationConfig(
            min_level_db=spec_config.min_level_db,
            ref_level_db=spec_config.ref_level_db,
        )

        sequence_length = spec.sequence_length
        evaluation_logger.info("Sequence length: {}".format(sequence_length))

        evaluation_logger.info("Evaluating on {} split".format(spec.split))
        results = _evaluate_dataset_split(
            model=model,
            dataset_handle=dataset_handle,
            split_name=spec.split,
            spec_config=spec_config,
            norm_config=norm_config,
            sequence_length=sequence_length,
            batch_size=spec.batch_size,
            num_workers=spec.num_workers,
            device=device,
            thresholds=spec.detection_thresholds,
            target_index=target_index,
        )

        evaluation_dir = run_dir / "outputs" / "evaluation"
        evaluation_report = {
            "run_id": run_state.run_id,
            "model_path": model_path,
            "dataset_name": spec.dataset_name,
            "sequence_length": sequence_length,
            "results": results,
        }
        write_json(evaluation_dir / "evaluation_report.json", evaluation_report)

        evaluation_logger.info(
            "Evaluation complete: accuracy={}, f1={}, auc={}".format(
                results["metrics"]["accuracy"],
                results["metrics"]["f1"],
                results["metrics"]["auc"],
            )
        )

        run_state = run_manager.mark_completed(run_state.run_id)
        return run_state

    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        logger.error("Evaluation run failed: {}".format(error_message))
        run_state = run_manager.mark_failed(run_state.run_id, error_message)
        raise
