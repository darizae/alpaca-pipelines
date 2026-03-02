"""
Training pipeline executor.

Bridges the run specification with bioacoustics-dl-toolbox components
to execute a full training run (train → validate → test → save model).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from bioacoustics_dl_toolbox.audio.datasets import SpectrogramDataset
from bioacoustics_dl_toolbox.config import (
    AugmentationConfig,
    ClassifierConfig,
    EncoderConfig,
    NormalizationConfig,
    SpectrogramConfig,
    TrainingConfig,
)
from bioacoustics_dl_toolbox.logging.logger import create_logger
from bioacoustics_dl_toolbox.metrics.core import Accuracy, F1Score, Precision, Recall
from bioacoustics_dl_toolbox.models.classifier import Classifier
from bioacoustics_dl_toolbox.models.encoder import ResidualEncoder
from bioacoustics_dl_toolbox.training.checkpoints import save_model
from bioacoustics_dl_toolbox.training.trainer import Trainer

from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.dataset.loader import DatasetHandle, load_dataset_handle
from alpaca_pipelines.runs.manager import RunManager
from alpaca_pipelines.training.config import TrainingRunSpec

logger = logging.getLogger(__name__)


def _build_spectrogram_config(spec: TrainingRunSpec) -> SpectrogramConfig:
    return SpectrogramConfig(
        sample_rate=spec.spectrogram.sample_rate,
        preemphasis=spec.spectrogram.preemphasis,
        n_fft=spec.spectrogram.n_fft,
        hop_length=spec.spectrogram.hop_length,
        n_freq_bins=spec.spectrogram.n_freq_bins,
        f_min=spec.spectrogram.f_min,
        f_max=spec.spectrogram.f_max,
        freq_compression=spec.spectrogram.freq_compression,
        min_level_db=spec.spectrogram.min_level_db,
        ref_level_db=spec.spectrogram.ref_level_db,
    )


def _build_normalization_config(spec: TrainingRunSpec) -> NormalizationConfig:
    return NormalizationConfig(
        mode=spec.normalization.mode,
        min_level_db=spec.normalization.min_level_db,
        ref_level_db=spec.normalization.ref_level_db,
    )


def _build_augmentation_config(spec: TrainingRunSpec) -> AugmentationConfig:
    return AugmentationConfig(
        enabled=spec.augmentation.enabled,
        amplitude_increase_db=spec.augmentation.amplitude_increase_db,
        amplitude_decrease_db=spec.augmentation.amplitude_decrease_db,
        time_stretch_from=spec.augmentation.time_stretch_from,
        time_stretch_to=spec.augmentation.time_stretch_to,
        pitch_shift_from=spec.augmentation.pitch_shift_from,
        pitch_shift_to=spec.augmentation.pitch_shift_to,
        noise_files=spec.augmentation.noise_files,
        noise_min_snr=spec.augmentation.noise_min_snr,
        noise_max_snr=spec.augmentation.noise_max_snr,
    )


def _build_encoder_config(spec: TrainingRunSpec) -> EncoderConfig:
    return EncoderConfig(
        input_channels=spec.encoder.input_channels,
        conv_kernel_size=spec.encoder.conv_kernel_size,
        max_pool=spec.encoder.max_pool,
        resnet_size=spec.encoder.resnet_size,
    )


def _build_classifier_config(spec: TrainingRunSpec, encoder: ResidualEncoder) -> ClassifierConfig:
    return ClassifierConfig(
        input_channels=encoder.output_channels,
        pooling=spec.classifier.pooling,
        num_classes=spec.classifier.num_classes,
    )


def _build_training_config(spec: TrainingRunSpec) -> TrainingConfig:
    return TrainingConfig(
        max_epochs=spec.training.max_epochs,
        batch_size=spec.training.batch_size,
        num_workers=spec.training.num_workers,
        learning_rate=spec.training.learning_rate,
        beta1=spec.training.beta1,
        lr_patience_epochs=spec.training.lr_patience_epochs,
        lr_decay_factor=spec.training.lr_decay_factor,
        early_stopping_patience_epochs=spec.training.early_stopping_patience_epochs,
        epochs_per_eval=spec.training.epochs_per_eval,
        val_metric=spec.training.val_metric,
        val_metric_mode=spec.training.val_metric_mode,
        use_cuda=spec.training.use_cuda,
        pin_memory=spec.training.pin_memory,
    )


def _create_dataset(
    file_names: list[str],
    dataset_handle: DatasetHandle,
    spec_config: SpectrogramConfig,
    norm_config: NormalizationConfig,
    aug_config: AugmentationConfig,
    sequence_length: int,
    positive_class: str,
    cache_dir: str | None,
    dataset_name: str,
) -> SpectrogramDataset:
    return SpectrogramDataset(
        file_names=file_names,
        working_dir=str(dataset_handle.snippets_dir),
        spec_config=spec_config,
        norm_config=norm_config,
        aug_config=aug_config,
        classes=dataset_handle.classes,
        sequence_length=sequence_length,
        class_to_index=dataset_handle.class_to_index,
        positive_class=positive_class,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
    )


def execute_training(
    run_state: RunState,
    environment: PipelineEnvironment,
    run_manager: RunManager,
) -> RunState:
    """Execute a training run from its persisted specification.

    This is the main entry point called by the CLI or SLURM job.
    It loads the dataset, builds the model, trains it, saves results,
    and updates the run state throughout.
    """
    spec = TrainingRunSpec.from_spec_dict(run_state.spec)
    run_dir = Path(run_state.run_dir)

    run_state = run_manager.mark_running(run_state.run_id)

    try:
        dataset_dir = environment.resolve_dataset_dir(spec.dataset_name)
        dataset_handle = load_dataset_handle(dataset_dir, environment.collection_root)

        spec_config = _build_spectrogram_config(spec)
        norm_config = _build_normalization_config(spec)
        aug_config_train = _build_augmentation_config(spec)
        aug_config_eval = AugmentationConfig(enabled=False)
        encoder_config = _build_encoder_config(spec)
        training_config = _build_training_config(spec)

        run_name = spec.run_name or run_state.run_id
        training_logger = create_logger(
            run_name,
            debug=False,
            log_dir=str(run_dir / "logs"),
        )

        training_logger.info("Loading dataset: {}".format(spec.dataset_name))
        training_logger.info(
            "Train: {} files, Val: {} files, Test: {} files".format(
                dataset_handle.split_file_count("train"),
                dataset_handle.split_file_count("val"),
                dataset_handle.split_file_count("test"),
            )
        )

        train_dataset = _create_dataset(
            file_names=dataset_handle.splits.train,
            dataset_handle=dataset_handle,
            spec_config=spec_config,
            norm_config=norm_config,
            aug_config=aug_config_train,
            sequence_length=spec.sequence_length,
            positive_class=spec.positive_class,
            cache_dir=spec.cache_dir,
            dataset_name="train",
        )

        val_dataset = _create_dataset(
            file_names=dataset_handle.splits.val,
            dataset_handle=dataset_handle,
            spec_config=spec_config,
            norm_config=norm_config,
            aug_config=aug_config_eval,
            sequence_length=spec.sequence_length,
            positive_class=spec.positive_class,
            cache_dir=spec.cache_dir,
            dataset_name="val",
        )

        test_dataset = _create_dataset(
            file_names=dataset_handle.splits.test,
            dataset_handle=dataset_handle,
            spec_config=spec_config,
            norm_config=norm_config,
            aug_config=aug_config_eval,
            sequence_length=spec.sequence_length,
            positive_class=spec.positive_class,
            cache_dir=spec.cache_dir,
            dataset_name="test",
        )

        device = torch.device(
            "cuda" if training_config.use_cuda and torch.cuda.is_available() else "cpu"
        )
        training_logger.info("Device: {}".format(device))

        encoder = ResidualEncoder(encoder_config)
        classifier_config = _build_classifier_config(spec, encoder)
        classifier = Classifier(classifier_config)
        model = nn.Sequential(encoder, classifier)

        training_logger.info(
            "Model: ResNet-{} encoder, {} classifier ({} classes)".format(
                encoder_config.resnet_size,
                classifier_config.pooling,
                classifier_config.num_classes,
            )
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory,
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=training_config.val_metric_mode,
            patience=training_config.lr_patience_epochs,
            factor=training_config.lr_decay_factor,
        )

        checkpoint_dir = str(run_dir / "outputs" / "model" / "checkpoints")
        summary_dir = str(run_dir / "outputs" / "summaries")

        trainer = Trainer(
            model=model,
            logger=training_logger,
            prefix=run_name,
            checkpoint_dir=checkpoint_dir,
            summary_dir=summary_dir,
            n_summaries=4,
            start_scratch=True,
        )

        metrics: dict[str, Any] = {
            "accuracy": Accuracy(device=device),
            "f1": F1Score(device=device),
            "precision": Precision(device=device),
            "recall": Recall(device=device),
        }

        run_manager.update_progress(
            run_state.run_id,
            total_epochs=training_config.max_epochs,
            current_epoch=0,
            current_phase="training",
        )

        model = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=training_config.max_epochs,
            val_interval=training_config.epochs_per_eval,
            patience_early_stopping=training_config.early_stopping_patience_epochs,
            device=device,
            metrics=metrics,
            val_metric=training_config.val_metric,
            val_metric_mode=training_config.val_metric_mode,
        )

        model_output_path = run_dir / "outputs" / "model" / "trained_model.pt"
        save_model(
            model=model,
            encoder=encoder,
            encoder_config=encoder_config,
            classifier=classifier,
            classifier_config=classifier_config,
            spec_config=spec_config,
            path=model_output_path,
            class_dist_dict=dataset_handle.class_to_index,
        )
        training_logger.info("Model saved to {}".format(model_output_path))

        run_manager.update_outputs(
            run_state.run_id,
            trained_model_path=str(model_output_path),
            tensorboard_dir=summary_dir,
        )

        run_state = run_manager.mark_completed(run_state.run_id)
        training_logger.info("Training run completed: {}".format(run_state.run_id))
        return run_state

    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        logger.error("Training run failed: {}".format(error_message))
        run_state = run_manager.mark_failed(run_state.run_id, error_message)
        raise
