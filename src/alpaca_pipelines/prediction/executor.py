"""
Prediction pipeline executor.

Supports sliding-window prediction over raw audio tapes
and batch inference on dataset test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from bioacoustics_dl_toolbox.audio.datasets import StridedAudioDataset
from bioacoustics_dl_toolbox.config import (
    ClassifierConfig,
    EncoderConfig,
    NormalizationConfig,
    SpectrogramConfig,
)
from bioacoustics_dl_toolbox.logging.logger import create_logger
from bioacoustics_dl_toolbox.models.classifier import Classifier
from bioacoustics_dl_toolbox.models.encoder import ResidualEncoder
from bioacoustics_dl_toolbox.training.checkpoints import load_model

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.io_utils import write_json
from alpaca_pipelines.prediction.config import PredictionRunSpec
from alpaca_pipelines.runs.manager import RunManager

logger = logging.getLogger(__name__)


def _validate_class_to_index(class_to_index: dict[str, int]) -> None:
    """Validate class mapping from model checkpoint.

    Ensures values form a contiguous range [0..N-1] and that
    "target" is present.
    """
    indices = sorted(class_to_index.values())
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            "Model class_to_index values are not contiguous [0..{}]: got {}".format(
                len(indices) - 1, indices
            )
        )
    if "target" not in class_to_index:
        raise ValueError(
            "Model class mapping does not contain 'target'. Available classes: {}".format(
                sorted(class_to_index.keys())
            )
        )


def _load_trained_model(
    model_path: str,
    device: torch.device,
) -> tuple[nn.Module, SpectrogramConfig, NormalizationConfig, dict[str, int]]:
    """Load a trained model from a checkpoint file."""
    model_dict = load_model(model_path)

    encoder_config = EncoderConfig(**model_dict["encoderConfig"])
    classifier_config = ClassifierConfig(**model_dict["classifierConfig"])
    spec_config = SpectrogramConfig(**model_dict["spectrogramConfig"])
    class_to_index: dict[str, int] = model_dict["classes"]
    _validate_class_to_index(class_to_index)

    encoder = ResidualEncoder(encoder_config)
    classifier = Classifier(classifier_config)
    encoder.load_state_dict(model_dict["encoderState"])
    classifier.load_state_dict(model_dict["classifierState"])

    model = nn.Sequential(encoder, classifier)
    model = model.to(device)
    model.eval()

    norm_config = NormalizationConfig(
        min_level_db=spec_config.min_level_db,
        ref_level_db=spec_config.ref_level_db,
    )

    return model, spec_config, norm_config, class_to_index


def _predict_tape(
    model: nn.Module,
    audio_file: str,
    spec_config: SpectrogramConfig,
    norm_config: NormalizationConfig,
    sequence_length_samples: int,
    hop_samples: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run sliding-window prediction over a single audio tape."""
    dataset = StridedAudioDataset(
        file_name=audio_file,
        sequence_len=sequence_length_samples,
        hop=hop_samples,
        spec_config=spec_config,
        norm_config=norm_config,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            probabilities = nn.functional.softmax(output, dim=1)
            all_scores.append(probabilities.cpu().numpy())

    if all_scores:
        scores_array = np.concatenate(all_scores, axis=0)
    else:
        scores_array = np.empty((0, 2))

    return {
        "audio_file": audio_file,
        "n_windows": len(dataset),
        "hop_samples": hop_samples,
        "sequence_length_samples": sequence_length_samples,
        "sample_rate": spec_config.sample_rate,
        "scores": scores_array.tolist(),
    }


def _generate_detections(
    tape_result: dict[str, Any],
    class_to_index: dict[str, int],
    detection_threshold: float,
    merge_overlapping: bool,
    min_detection_duration_s: float,
) -> list[dict[str, Any]]:
    """Convert window-level scores to time-aligned detections."""
    target_index = class_to_index["target"]
    scores = tape_result["scores"]
    hop_samples = tape_result["hop_samples"]
    sample_rate = tape_result["sample_rate"]
    sequence_length_samples = tape_result["sequence_length_samples"]
    hop_seconds = hop_samples / sample_rate
    window_duration_seconds = sequence_length_samples / sample_rate

    raw_detections: list[dict[str, Any]] = []
    for window_index, score_row in enumerate(scores):
        target_score = float(score_row[target_index])
        if target_score >= detection_threshold:
            start_seconds = window_index * hop_seconds
            end_seconds = start_seconds + window_duration_seconds
            raw_detections.append(
                {
                    "start_s": round(start_seconds, 4),
                    "end_s": round(end_seconds, 4),
                    "score": round(target_score, 6),
                }
            )

    if not raw_detections:
        return []

    if merge_overlapping and len(raw_detections) > 1:
        merged: list[dict[str, Any]] = [raw_detections[0].copy()]
        for detection in raw_detections[1:]:
            previous = merged[-1]
            if detection["start_s"] <= previous["end_s"]:
                previous["end_s"] = max(previous["end_s"], detection["end_s"])
                previous["score"] = max(previous["score"], detection["score"])
            else:
                merged.append(detection.copy())
        raw_detections = merged

    if min_detection_duration_s > 0.0:
        raw_detections = [
            detection
            for detection in raw_detections
            if (detection["end_s"] - detection["start_s"]) >= min_detection_duration_s
        ]

    return raw_detections


def execute_prediction(
    run_state: RunState,
    environment: PipelineEnvironment,
    run_manager: RunManager,
) -> RunState:
    """Execute a prediction run from its persisted specification."""
    spec = PredictionRunSpec.from_spec_dict(run_state.spec)
    run_dir = Path(run_state.run_dir)

    run_state = run_manager.mark_running(run_state.run_id)

    try:
        run_name = spec.run_name or run_state.run_id
        prediction_logger = create_logger(
            run_name,
            debug=False,
            log_dir=str(run_dir / "logs"),
        )

        device = torch.device("cuda" if spec.use_cuda and torch.cuda.is_available() else "cpu")
        prediction_logger.info("Device: {}".format(device))

        prediction_logger.info("Loading model from: {}".format(spec.model_path))
        model, spec_config, norm_config, class_to_index = _load_trained_model(
            spec.model_path, device
        )

        if spec.normalization.mode != "db":
            norm_config = NormalizationConfig(
                mode=spec.normalization.mode,
                min_level_db=spec.normalization.min_level_db,
                ref_level_db=spec.normalization.ref_level_db,
            )

        sequence_length_samples = int(spec.sequence_length_ms / 1000.0 * spec_config.sample_rate)
        hop_samples = int(spec.hop_ms / 1000.0 * spec_config.sample_rate)

        audio_files: list[str] = []
        if spec.mode == "tape":
            audio_files = list(spec.audio_files)
        elif spec.mode == "dataset":
            if spec.dataset_name is None:
                raise ValueError("Prediction mode is 'dataset' but dataset_name is not set")
            from alpaca_pipelines.dataset.loader import load_dataset_handle

            dataset_dir = environment.resolve_dataset_dir(spec.dataset_name)
            dataset_handle = load_dataset_handle(dataset_dir)
            for filename in dataset_handle.splits.test:
                audio_files.append(str(dataset_handle.snippets_dir / filename))

        if not audio_files:
            raise ValueError("No audio files to predict")

        predictions_dir = run_dir / "outputs" / "predictions"
        all_results: list[dict[str, Any]] = []

        for file_index, audio_file in enumerate(audio_files):
            prediction_logger.info(
                "Predicting [{}/{}]: {}".format(file_index + 1, len(audio_files), audio_file)
            )

            tape_result = _predict_tape(
                model=model,
                audio_file=audio_file,
                spec_config=spec_config,
                norm_config=norm_config,
                sequence_length_samples=sequence_length_samples,
                hop_samples=hop_samples,
                batch_size=spec.batch_size,
                num_workers=spec.num_workers,
                device=device,
            )

            detections = _generate_detections(
                tape_result=tape_result,
                class_to_index=class_to_index,
                detection_threshold=spec.detection_threshold,
                merge_overlapping=spec.merge_overlapping,
                min_detection_duration_s=spec.min_detection_duration_s,
            )

            file_result = {
                "audio_file": audio_file,
                "n_windows": tape_result["n_windows"],
                "n_detections": len(detections),
                "detections": detections,
                "scores_shape": [len(tape_result["scores"]), len(tape_result["scores"][0])]
                if tape_result["scores"]
                else [0, 0],
            }
            all_results.append(file_result)

            per_file_path = predictions_dir / "{}.json".format(Path(audio_file).stem)
            write_json(per_file_path, file_result)

            run_manager.update_progress(
                run_state.run_id,
                current_phase="prediction",
                current_epoch=file_index + 1,
                total_epochs=len(audio_files),
            )

        summary = {
            "run_id": run_state.run_id,
            "model_path": spec.model_path,
            "n_files": len(audio_files),
            "total_detections": sum(r["n_detections"] for r in all_results),
            "detection_threshold": spec.detection_threshold,
            "files": [
                {
                    "audio_file": r["audio_file"],
                    "n_windows": r["n_windows"],
                    "n_detections": r["n_detections"],
                }
                for r in all_results
            ],
        }
        write_json(predictions_dir / "prediction_summary.json", summary)

        prediction_logger.info(
            "Prediction complete: {} files, {} total detections".format(
                summary["n_files"], summary["total_detections"]
            )
        )

        if spec.apply_rf_filter and spec.rf_model_path is not None:
            prediction_logger.info("Applying RF filter (post-processing)")
            from alpaca_pipelines.rf.executor import apply_rf_filter

            apply_rf_filter(
                predictions_dir=predictions_dir,
                rf_model_path=spec.rf_model_path,
                audio_files=audio_files,
                sample_rate=spec_config.sample_rate,
                environment=environment,
                prediction_logger=prediction_logger,
            )

            run_manager.update_outputs(
                run_state.run_id,
                rf_filtered=True,
            )

        run_state = run_manager.mark_completed(run_state.run_id)
        return run_state

    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        logger.error("Prediction run failed: {}".format(error_message))
        run_state = run_manager.mark_failed(run_state.run_id, error_message)
        raise
