"""
Random Forest filter post-processing.

Applies an RF classifier to CNN detections, re-scoring each detection
based on acoustic features extracted from the detection window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import soundfile as sf
from bioacoustics_dl_toolbox.rf.features import compute_rf_features
from bioacoustics_dl_toolbox.rf.types import RfClassifierProtocol

from alpaca_pipelines.io_utils import read_json, write_json


def _load_rf_model(rf_model_path: str) -> RfClassifierProtocol:
    """Load a serialized RF model from disk."""
    model: RfClassifierProtocol = joblib.load(rf_model_path)
    return model


def _load_audio_signal(audio_file: str) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 signal."""
    audio_data, sample_rate = sf.read(audio_file, always_2d=True, dtype="float32")
    if audio_data.ndim == 2 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    else:
        audio_data = audio_data.flatten()
    return audio_data, int(sample_rate)


def apply_rf_filter(
    predictions_dir: Path,
    rf_model_path: str,
    audio_files: list[str],
    prediction_logger: logging.Logger,
) -> None:
    """Apply RF filter to all prediction files in the predictions directory.

    For each detection, extract acoustic features from the detection window
    and re-score using the RF model.  Writes filtered results alongside
    the original predictions.
    """
    rf_model = _load_rf_model(rf_model_path)
    feature_names = getattr(rf_model, "feature_names_in_", None)

    prediction_logger.info("RF model loaded from: {}".format(rf_model_path))

    for audio_file in audio_files:
        prediction_file = predictions_dir / "{}.json".format(Path(audio_file).stem)
        if not prediction_file.is_file():
            prediction_logger.warning(
                "No prediction file for {}, skipping RF filter".format(audio_file)
            )
            continue

        prediction_data = read_json(prediction_file)
        detections = prediction_data.get("detections", [])
        if not detections:
            continue

        signal, file_sample_rate = _load_audio_signal(audio_file)

        filtered_detections: list[dict[str, Any]] = []
        for detection in detections:
            start_s = float(detection["start_s"])
            end_s = float(detection["end_s"])

            features = compute_rf_features(
                signal=signal,
                sample_rate=file_sample_rate,
                start_s=start_s,
                end_s=end_s,
            )

            if feature_names is not None:
                feature_vector = np.array(
                    [features[name] for name in feature_names], dtype=np.float64
                ).reshape(1, -1)
            else:
                feature_vector = np.array(list(features.values()), dtype=np.float64).reshape(1, -1)

            if np.any(~np.isfinite(feature_vector)):
                detection["rf_score"] = None
                detection["rf_pass"] = False
            else:
                rf_probabilities = rf_model.predict_proba(feature_vector)
                rf_target_score = float(rf_probabilities[0, 1])
                detection["rf_score"] = round(rf_target_score, 6)
                detection["rf_pass"] = rf_target_score >= 0.5

            filtered_detections.append(detection)

        filtered_data = dict(prediction_data)
        filtered_data["detections"] = filtered_detections
        filtered_data["rf_filtered"] = True
        filtered_data["rf_model_path"] = rf_model_path

        filtered_path = predictions_dir / "{}_rf_filtered.json".format(Path(audio_file).stem)
        write_json(filtered_path, filtered_data)

        n_passed = sum(1 for d in filtered_detections if d.get("rf_pass", False))
        prediction_logger.info(
            "RF filter {}: {}/{} detections passed".format(
                Path(audio_file).name, n_passed, len(filtered_detections)
            )
        )
