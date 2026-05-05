"""Random Forest filter post-processing."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import soundfile as sf

from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.rf.audio_features import mfcc_summary, raven_robust_features
from alpaca_pipelines.rf.audio_preprocess import prepare_rf_segment
from alpaca_pipelines.rf.config import RfFeatureConfig
from bioacoustics_dl_toolbox.rf.types import RfClassifierProtocol


def _load_rf_model(rf_model_path: str) -> RfClassifierProtocol:
    model: RfClassifierProtocol = joblib.load(rf_model_path)
    return model


def _load_rf_model_metadata(rf_model_path: str) -> dict[str, Any]:
    metadata_path = Path(rf_model_path).with_name("rf_model_metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError("RF model metadata file missing: {}".format(metadata_path))

    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("RF model metadata must be a JSON object: {}".format(metadata_path))
    return metadata


def _validate_rf_metadata_contract(metadata: dict[str, Any]) -> None:
    feature_family = metadata.get("feature_family")
    if feature_family is None:
        raise ValueError("RF model metadata missing required field 'feature_family'")
    if feature_family != "rf_v1":
        raise ValueError(
            "Unsupported RF model feature_family {!r}; expected 'rf_v1'".format(feature_family)
        )


def _validate_no_legacy_cnn_feature(feature_names: np.ndarray | None) -> None:
    if feature_names is None:
        return

    if any(str(name) == "cnn_logit_mean" for name in feature_names):
        raise ValueError(
            "Unsupported RF model: legacy feature 'cnn_logit_mean' is not supported by rf_v1"
        )


def _read_audio_segment(
    audio_handle: sf.SoundFile,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, int]:
    sample_rate = int(audio_handle.samplerate)
    total_frames = len(audio_handle)
    if sample_rate <= 0 or total_frames <= 0:
        return np.zeros(0, dtype=np.float32), sample_rate

    if not math.isfinite(start_s):
        start_s = 0.0
    if not math.isfinite(end_s):
        end_s = 0.0

    start_frame = int(round(max(0.0, start_s) * float(sample_rate)))
    end_frame = int(round(max(0.0, end_s) * float(sample_rate)))
    start_frame = min(max(0, start_frame), total_frames)
    end_frame = min(max(0, end_frame), total_frames)
    if end_frame <= start_frame:
        return np.zeros(0, dtype=np.float32), sample_rate

    audio_handle.seek(start_frame)
    audio_data = audio_handle.read(end_frame - start_frame, always_2d=True, dtype="float32")
    if audio_data.size == 0:
        return np.zeros(0, dtype=np.float32), sample_rate
    if audio_data.shape[1] > 1:
        mono = np.mean(audio_data, axis=1, dtype=np.float32)
    else:
        mono = audio_data[:, 0]
    return np.asarray(mono, dtype=np.float32), sample_rate


def _resolve_feature_config(
    rf_feature_config: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> RfFeatureConfig:
    if rf_feature_config is not None:
        return RfFeatureConfig.model_validate(rf_feature_config)

    metadata_feature_config = metadata.get("feature_config")
    if metadata_feature_config is None:
        raise ValueError(
            "rf_feature_config is required when RF model metadata does not include feature_config"
        )
    return RfFeatureConfig.model_validate(metadata_feature_config)


def apply_rf_filter(
    prediction_inputs: list[dict[str, str]],
    rf_model_path: str,
    rf_threshold: float,
    rf_feature_config: dict[str, Any] | None,
    prediction_logger: logging.Logger,
) -> dict[str, Any]:
    rf_model = _load_rf_model(rf_model_path)
    metadata = _load_rf_model_metadata(rf_model_path)
    _validate_rf_metadata_contract(metadata)
    active_feature_config = _resolve_feature_config(
        rf_feature_config=rf_feature_config,
        metadata=metadata,
    )
    feature_names = getattr(rf_model, "feature_names_in_", None)

    if feature_names is None:
        metadata_feature_names = metadata.get("feature_names")
        if metadata_feature_names is not None:
            feature_names = np.asarray(metadata_feature_names, dtype=object)
    _validate_no_legacy_cnn_feature(feature_names=feature_names)

    prediction_logger.info("RF model loaded from: {}".format(rf_model_path))

    file_summaries: list[dict[str, Any]] = []
    total_base = 0
    total_passed = 0
    total_rejected = 0
    total_unscored = 0

    for item in prediction_inputs:
        audio_file = item["audio_file"]
        prediction_file = Path(item["prediction_file"])

        if not prediction_file.is_file():
            prediction_logger.warning(
                "No prediction file for {}, skipping RF filter".format(audio_file)
            )
            continue

        prediction_data = read_json(prediction_file)
        detections = prediction_data.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError(
                "Prediction payload must contain a list of detections: {}".format(prediction_file)
            )
        if not detections:
            filtered_data = dict(prediction_data)
            filtered_data["rf_filtered"] = True
            filtered_data["rf_model_path"] = rf_model_path
            filtered_data["rf_threshold"] = rf_threshold
            filtered_path = prediction_file.with_name(f"{prediction_file.stem}_rf_filtered.json")
            write_json(filtered_path, filtered_data)
            file_summaries.append(
                {
                    "audio_file": audio_file,
                    "prediction_file": str(prediction_file),
                    "rf_filtered_file": str(filtered_path),
                    "base_detections": 0,
                    "rf_passed": 0,
                    "rf_rejected": 0,
                    "rf_unscored": 0,
                    "rejection_rate": 0.0,
                    "pass_rate": 0.0,
                }
            )
            prediction_logger.info(
                "RF filter {}: 0/0 detections passed".format(Path(audio_file).name)
            )
            continue

        filtered_detections: list[dict[str, Any]] = []
        with sf.SoundFile(audio_file) as source_audio:
            for detection in detections:
                start_s = float(detection["start_s"])
                end_s = float(detection["end_s"])
                raw_segment, file_sample_rate = _read_audio_segment(
                    source_audio,
                    start_s=start_s,
                    end_s=end_s,
                )
                segment_duration_s = float(raw_segment.shape[0]) / float(file_sample_rate or 1)
                segment, rf_sr = prepare_rf_segment(
                    signal=raw_segment,
                    source_sr=file_sample_rate,
                    t0=0.0,
                    t1=segment_duration_s,
                    config=active_feature_config,
                )

                robust = raven_robust_features(
                    y=segment,
                    sr=rf_sr,
                    fmin=active_feature_config.fmin_hz,
                    fmax=active_feature_config.fmax_hz,
                    n_fft=active_feature_config.n_fft,
                    hop_length=active_feature_config.hop_length,
                )
                mfcc = mfcc_summary(
                    y=segment,
                    sr=rf_sr,
                    n_mfcc=active_feature_config.n_mfcc,
                    n_fft=active_feature_config.n_fft,
                    hop_length=active_feature_config.hop_length,
                    include_deltas=active_feature_config.include_deltas,
                )
                feature_row = {**robust, **mfcc}

                if feature_names is not None:
                    try:
                        feature_vector = np.array(
                            [float(feature_row[str(name)]) for name in feature_names],
                            dtype=np.float64,
                        ).reshape(1, -1)
                    except KeyError as exc:
                        raise KeyError(
                            "Missing required RF feature '{}' for detection in {}".format(
                                exc.args[0], audio_file
                            )
                        ) from exc
                else:
                    feature_vector = np.array(
                        list(feature_row.values()),
                        dtype=np.float64,
                    ).reshape(1, -1)

                detection_with_rf = dict(detection)
                if np.any(~np.isfinite(feature_vector)):
                    detection_with_rf["rf_score"] = None
                    detection_with_rf["rf_pass"] = False
                else:
                    rf_probabilities = rf_model.predict_proba(feature_vector)
                    rf_target_score = float(rf_probabilities[0, 1])
                    detection_with_rf["rf_score"] = round(rf_target_score, 6)
                    detection_with_rf["rf_pass"] = rf_target_score >= rf_threshold

                filtered_detections.append(detection_with_rf)

        filtered_data = dict(prediction_data)
        filtered_data["detections"] = filtered_detections
        filtered_data["rf_filtered"] = True
        filtered_data["rf_model_path"] = rf_model_path
        filtered_data["rf_threshold"] = rf_threshold

        filtered_path = prediction_file.with_name(f"{prediction_file.stem}_rf_filtered.json")
        write_json(filtered_path, filtered_data)

        n_passed = sum(1 for d in filtered_detections if d.get("rf_pass", False))
        n_unscored = sum(1 for d in filtered_detections if d.get("rf_score") is None)
        n_rejected = sum(
            1
            for d in filtered_detections
            if not d.get("rf_pass", False) and d.get("rf_score") is not None
        )
        base_detections = len(filtered_detections)
        total_base += base_detections
        total_passed += n_passed
        total_rejected += n_rejected
        total_unscored += n_unscored
        file_summaries.append(
            {
                "audio_file": audio_file,
                "prediction_file": str(prediction_file),
                "rf_filtered_file": str(filtered_path),
                "base_detections": base_detections,
                "rf_passed": n_passed,
                "rf_rejected": n_rejected,
                "rf_unscored": n_unscored,
                "rejection_rate": round(n_rejected / base_detections, 6)
                if base_detections
                else 0.0,
                "pass_rate": round(n_passed / base_detections, 6) if base_detections else 0.0,
            }
        )
        prediction_logger.info(
            "RF filter {}: {}/{} detections passed".format(
                Path(audio_file).name, n_passed, len(filtered_detections)
            )
        )

    return {
        "applied": True,
        "rf_model_path": rf_model_path,
        "rf_threshold": rf_threshold,
        "base_detections": total_base,
        "rf_passed": total_passed,
        "rf_rejected": total_rejected,
        "rf_unscored": total_unscored,
        "rejection_rate": round(total_rejected / total_base, 6) if total_base else 0.0,
        "pass_rate": round(total_passed / total_base, 6) if total_base else 0.0,
        "files": file_summaries,
    }
