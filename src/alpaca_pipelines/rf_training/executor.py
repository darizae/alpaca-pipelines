"""RF training pipeline executor."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.dataset.loader import DatasetHandle, load_dataset_handle
from alpaca_pipelines.io_utils import write_json
from alpaca_pipelines.rf.audio_features import mfcc_summary, raven_robust_features
from alpaca_pipelines.rf.audio_preprocess import prepare_rf_segment
from alpaca_pipelines.rf.config import RfFeatureConfig
from alpaca_pipelines.rf_training.config import RfTrainingRunSpec
from alpaca_pipelines.runs.manager import RunManager
from bioacoustics_dl_toolbox.logging.logger import create_logger

logger = logging.getLogger(__name__)


def _load_audio_signal(audio_path: Path) -> tuple[np.ndarray, int]:
    audio_data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    if audio_data.ndim != 2:
        raise ValueError(
            "Expected always_2d=True to return a 2D array, got ndim={}".format(audio_data.ndim)
        )
    if audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    else:
        audio_data = audio_data[:, 0]
    return audio_data, int(sample_rate)


def _label_for_filename(
    dataset_handle: DatasetHandle,
    filename: str,
    positive_class: str,
) -> int:
    if positive_class not in dataset_handle.class_to_index:
        raise ValueError(
            "positive_class {!r} not found in dataset classes: {}".format(
                positive_class, dataset_handle.classes
            )
        )

    snippet = next((s for s in dataset_handle.manifest.snippets if s.filename == filename), None)
    if snippet is None:
        raise ValueError("Filename not found in manifest: {}".format(filename))

    if snippet.classification == positive_class:
        return 1
    return 0


def _compute_features_for_file(
    audio_path: Path,
    feature_config: RfFeatureConfig,
) -> dict[str, float]:
    signal, source_sr = _load_audio_signal(audio_path)
    duration_s = float(len(signal)) / float(source_sr)
    segment, rf_sr = prepare_rf_segment(
        signal=signal,
        source_sr=source_sr,
        t0=0.0,
        t1=duration_s,
        config=feature_config,
    )
    robust = raven_robust_features(
        y=segment,
        sr=rf_sr,
        fmin=feature_config.fmin_hz,
        fmax=feature_config.fmax_hz,
        n_fft=feature_config.n_fft,
        hop_length=feature_config.hop_length,
    )
    mfcc = mfcc_summary(
        y=segment,
        sr=rf_sr,
        n_mfcc=feature_config.n_mfcc,
        n_fft=feature_config.n_fft,
        hop_length=feature_config.hop_length,
        include_deltas=feature_config.include_deltas,
    )
    features = {**robust, **mfcc}

    feature_values = np.array(list(features.values()), dtype=np.float64)
    if np.any(~np.isfinite(feature_values)):
        non_finite_names = [k for k, v in features.items() if not np.isfinite(float(v))]
        raise ValueError(
            "Non-finite RF features for {}: {}".format(audio_path.name, ", ".join(non_finite_names))
        )

    return {str(k): float(v) for k, v in features.items()}


def _build_feature_table(
    dataset_handle: DatasetHandle,
    filenames: list[str],
    positive_class: str,
    feature_config: RfFeatureConfig,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    used_files: list[str] = []

    for filename in filenames:
        audio_path = dataset_handle.snippets_dir / filename
        if not audio_path.is_file():
            raise FileNotFoundError("Snippet file missing: {}".format(audio_path))

        features = _compute_features_for_file(audio_path, feature_config=feature_config)
        label = _label_for_filename(dataset_handle, filename, positive_class)

        rows.append(features)
        labels.append(label)
        used_files.append(filename)

    if not rows:
        raise ValueError("No files provided to build feature table")

    table = pd.DataFrame(rows)
    if table.isnull().any().any():
        missing_columns = sorted([c for c in table.columns if table[c].isnull().any()])
        raise ValueError(
            "RF feature table contains NaNs in columns: {}".format(", ".join(missing_columns))
        )

    labels_array = np.array(labels, dtype=np.int64)
    return table, labels_array, used_files


def execute_rf_training(
    run_state: RunState,
    environment: PipelineEnvironment,
    run_manager: RunManager,
) -> RunState:
    """Execute an RF training run from its persisted specification."""
    spec = RfTrainingRunSpec.from_spec_dict(run_state.spec)
    run_dir = Path(run_state.run_dir)

    run_state = run_manager.mark_running(run_state.run_id)

    try:
        run_name = spec.run_name or run_state.run_id
        rf_logger = create_logger(
            run_name,
            debug=False,
            log_dir=str(run_dir / "logs"),
        )

        dataset_dir = environment.resolve_dataset_dir(spec.dataset_name)
        dataset_handle = load_dataset_handle(dataset_dir, environment.collection_root)

        train_files = list(dataset_handle.splits.train)
        val_files = list(dataset_handle.splits.val)

        if not train_files:
            raise ValueError("RF training requires a non-empty train split")
        if not val_files:
            raise ValueError("RF training requires a non-empty val split")

        rf_logger.info("Loading dataset: {}".format(spec.dataset_name))
        rf_logger.info("Train: {} files, Val: {} files".format(len(train_files), len(val_files)))

        run_manager.update_progress(
            run_state.run_id,
            current_phase="feature_extraction",
            current_epoch=0,
            total_epochs=3,
        )

        x_train, y_train, used_train_files = _build_feature_table(
            dataset_handle=dataset_handle,
            filenames=train_files,
            positive_class=spec.positive_class,
            feature_config=spec.feature_config,
        )
        x_val, y_val, used_val_files = _build_feature_table(
            dataset_handle=dataset_handle,
            filenames=val_files,
            positive_class=spec.positive_class,
            feature_config=spec.feature_config,
        )

        rf_logger.info("RF features: {} columns".format(x_train.shape[1]))

        run_manager.update_progress(
            run_state.run_id,
            current_phase="training",
            current_epoch=1,
            total_epochs=3,
        )

        model = RandomForestClassifier(
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            min_samples_split=spec.min_samples_split,
            min_samples_leaf=spec.min_samples_leaf,
            max_features=spec.max_features,
            class_weight=spec.class_weight,
            n_jobs=spec.n_jobs,
            random_state=spec.random_state,
        )

        model.fit(x_train, y_train)

        run_manager.update_progress(
            run_state.run_id,
            current_phase="validation",
            current_epoch=2,
            total_epochs=3,
        )

        val_probabilities = model.predict_proba(x_val)
        val_predictions = (val_probabilities[:, 1] >= spec.rf_threshold).astype(np.int64)

        accuracy = float(accuracy_score(y_val, val_predictions))
        f1 = float(f1_score(y_val, val_predictions, zero_division=0))
        precision = float(precision_score(y_val, val_predictions, zero_division=0))
        recall = float(recall_score(y_val, val_predictions, zero_division=0))

        rf_logger.info(
            "Validation metrics: accuracy={}, f1={}, precision={}, recall={}".format(
                round(accuracy, 6),
                round(f1, 6),
                round(precision, 6),
                round(recall, 6),
            )
        )

        model_dir = run_dir / "outputs" / "model"
        model_path = model_dir / "rf_model.joblib"
        if model_path.exists():
            raise FileExistsError("RF model output already exists: {}".format(model_path))

        joblib.dump(model, model_path)
        feature_names = list(map(str, x_train.columns.tolist()))
        metadata = {
            "feature_family": "rf_v1",
            "feature_names": feature_names,
            "rf_threshold": spec.rf_threshold,
            "feature_config": spec.feature_config.model_dump(),
        }
        write_json(model_dir / "rf_model_metadata.json", metadata)

        report = {
            "run_id": run_state.run_id,
            "dataset_name": spec.dataset_name,
            "positive_class": spec.positive_class,
            "class_to_index": dataset_handle.class_to_index,
            "train": {
                "n_samples": int(len(y_train)),
                "n_positive": int(np.sum(y_train)),
                "n_negative": int(len(y_train) - np.sum(y_train)),
                "files": used_train_files,
            },
            "val": {
                "n_samples": int(len(y_val)),
                "n_positive": int(np.sum(y_val)),
                "n_negative": int(len(y_val) - np.sum(y_val)),
                "files": used_val_files,
            },
            "features": {
                "n_features": int(x_train.shape[1]),
                "feature_names": feature_names,
            },
            "feature_family": "rf_v1",
            "rf_threshold": spec.rf_threshold,
            "feature_config": spec.feature_config.model_dump(),
            "hyperparameters": {
                "n_estimators": spec.n_estimators,
                "max_depth": spec.max_depth,
                "min_samples_split": spec.min_samples_split,
                "min_samples_leaf": spec.min_samples_leaf,
                "max_features": spec.max_features,
                "class_weight": spec.class_weight,
                "n_jobs": spec.n_jobs,
                "random_state": spec.random_state,
            },
            "metrics": {
                "accuracy": round(accuracy, 6),
                "f1": round(f1, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "classification_report": classification_report(
                    y_val, val_predictions, output_dict=True, zero_division=0
                ),
            },
            "model_path": str(model_path),
        }

        summaries_dir = run_dir / "outputs" / "summaries"
        write_json(summaries_dir / "rf_training_report.json", report)

        run_manager.update_outputs(
            run_state.run_id,
            rf_model_path=str(model_path),
        )

        run_manager.update_progress(
            run_state.run_id,
            current_phase="completed",
            current_epoch=3,
            total_epochs=3,
            best_metric_name="f1",
            best_metric_value=f1,
        )

        run_state = run_manager.mark_completed(run_state.run_id)
        rf_logger.info("RF training run completed: {}".format(run_state.run_id))
        return run_state

    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        logger.error("RF training run failed: {}".format(error_message))
        run_state = run_manager.mark_failed(run_state.run_id, error_message)
        raise
