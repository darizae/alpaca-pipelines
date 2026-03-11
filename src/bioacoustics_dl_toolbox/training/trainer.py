"""
Training loop with validation, testing, checkpointing, and TensorBoard logging.

Ported from ANIMAL-SPOT ``trainer.py`` (Bergler & Schroeter, GPL-3.0).
Key changes:
- Accepts an optional ``SummaryWriter``; does not create one internally.
- Uses standard ``logging.Logger`` instead of Singleton Logger.
- Visualization helpers are called as standalone functions.
"""

from __future__ import annotations

import copy
import logging
import math
import operator
import platform
import time
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn

from bioacoustics_dl_toolbox.metrics.auc import AUCMeter
from bioacoustics_dl_toolbox.metrics.confusion import ConfusionMeter
from bioacoustics_dl_toolbox.metrics.core import Mean, MetricBase, Sum
from bioacoustics_dl_toolbox.training.checkpoints import CheckpointHandler
from bioacoustics_dl_toolbox.training.early_stopping import EarlyStoppingCriterion
from bioacoustics_dl_toolbox.visualization.image import prepare_img
from bioacoustics_dl_toolbox.visualization.plotting import confusion_matrix_fig, roc_fig

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment, misc]


class Trainer:
    """Manages the train/val/test loop with checkpointing and logging.

    Parameters
    ----------
    model:
        The full ``nn.Sequential(encoder, classifier)`` model.
    logger:
        A ``logging.Logger`` instance.
    prefix:
        Name prefix for checkpoints and TensorBoard runs.
    checkpoint_dir:
        Directory for saving checkpoints. ``None`` to disable.
    summary_dir:
        Directory for TensorBoard summaries. ``None`` to disable.
    n_summaries:
        Number of sample images to write per epoch.
    start_scratch:
        If ``True``, ignore existing checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        logger: logging.Logger,
        prefix: str = "",
        checkpoint_dir: str | None = None,
        summary_dir: str | None = None,
        n_summaries: int = 4,
        input_shape: tuple[int, ...] | None = None,
        start_scratch: bool = False,
    ) -> None:
        self.model = model
        self.logger = logger
        self.prefix = prefix
        self.n_summaries = n_summaries

        self.logger.info("Init summary writer")
        if summary_dir is not None and SummaryWriter is not None:
            run_name = prefix + "_" if prefix != "" else ""
            run_name += "{time}-{host}".format(
                time=time.strftime("%y-%m-%d-%H-%M", time.localtime()),
                host=platform.uname()[1],
            )
            import os

            summary_dir = os.path.join(summary_dir, run_name)
            self.writer: Any = SummaryWriter(summary_dir)
            if input_shape is not None:
                dummy_input = torch.rand(input_shape)
                self.logger.info("Writing graph to summary")
                self.writer.add_graph(self.model, dummy_input)
        else:
            self.writer = None

        if checkpoint_dir is not None:
            self.checkpoint_handler: CheckpointHandler | None = CheckpointHandler(
                checkpoint_dir, prefix=prefix, logger=self.logger
            )
        else:
            self.checkpoint_handler = None

        self.start_scratch = start_scratch
        self.class_dist_dict: dict[str, int] | None = None

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        n_epochs: int,
        val_interval: int,
        patience_early_stopping: int,
        device: torch.device,
        metrics: dict[str, MetricBase] | list[MetricBase] = {},
        val_metric: str = "loss",
        val_metric_mode: str = "min",
        start_epoch: int = 0,
    ) -> nn.Module:
        """Run the full train → validate → test pipeline.

        Returns the model loaded with the best validation weights.
        """
        self.logger.info("Init model on device '{}'".format(device))
        self.model = self.model.to(device)
        self.class_dist_dict = train_loader.dataset.class_dist_dict  # type: ignore[union-attr]

        best_model = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0 if val_metric_mode == "max" else float("inf")

        patience_stopping = math.ceil(patience_early_stopping / val_interval)
        patience_stopping = int(max(1, patience_stopping))
        early_stopping = EarlyStoppingCriterion(
            mode=val_metric_mode, patience=patience_stopping  # type: ignore[arg-type]
        )

        if not self.start_scratch and self.checkpoint_handler is not None:
            checkpoint = self.checkpoint_handler.read_latest()
            if checkpoint is not None:
                try:
                    try:
                        self.model.load_state_dict(checkpoint["modelState"])
                    except RuntimeError as e:
                        self.logger.error(
                            "Failed to restore checkpoint: "
                            "Checkpoint has different parameters"
                        )
                        self.logger.error(str(e))
                        raise SystemExit
                    optimizer.load_state_dict(checkpoint["trainState"]["optState"])
                    start_epoch = checkpoint["trainState"]["epoch"] + 1
                    best_metric = checkpoint["trainState"]["best_metric"]
                    best_model = checkpoint["trainState"]["best_model"]
                    early_stopping.load_state_dict(
                        checkpoint["trainState"]["earlyStopping"]
                    )
                    scheduler.load_state_dict(checkpoint["trainState"]["scheduler"])
                    self.logger.info("Resuming with epoch {}".format(start_epoch))
                except KeyError:
                    self.logger.error("Failed to restore checkpoint")
                    raise

        since = time.time()
        self.logger.info("Class Distribution: " + str(self.class_dist_dict))
        self.logger.info("Start training model " + self.prefix)

        try:
            val_comp = operator.gt if val_metric_mode == "max" else operator.lt

            for epoch in range(start_epoch, n_epochs):
                self._train_epoch(
                    epoch, train_loader, loss_fn, optimizer, metrics, device
                )
                if epoch % val_interval == 0 or epoch == n_epochs - 1:
                    val_loss = self._test_epoch(
                        epoch, val_loader, loss_fn, metrics, device, phase="val"
                    )
                    if val_metric == "loss":
                        val_result = val_loss
                    else:
                        val_result = metrics[val_metric].get()  # type: ignore[index]

                    if val_comp(val_result, best_metric):
                        best_metric = val_result
                        best_model = copy.deepcopy(self.model.state_dict())

                    if self.checkpoint_handler is not None:
                        self.checkpoint_handler.write(
                            {
                                "modelState": self.model.state_dict(),
                                "trainState": {
                                    "epoch": epoch,
                                    "best_metric": best_metric,
                                    "best_model": best_model,
                                    "optState": optimizer.state_dict(),
                                    "earlyStopping": early_stopping.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                },
                                "classes": self.class_dist_dict,
                            }
                        )

                    scheduler.step(val_result)

                    if early_stopping.step(val_result):
                        self.logger.info(
                            "No improvement over the last {} epochs. Stopping.".format(
                                patience_early_stopping
                            )
                        )
                        break
        except Exception:
            import traceback

            self.logger.warning(traceback.format_exc())
            self.logger.warning("Aborting...")
            raise SystemExit

        self.model.load_state_dict(best_model)
        final_loss = self._test_epoch(
            0, test_loader, loss_fn, metrics, device, phase="test"
        )
        if val_metric == "loss":
            final_metric = final_loss
        else:
            final_metric = metrics[val_metric].get()  # type: ignore[index]

        time_elapsed = time.time() - since
        self.logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        self.logger.info("Best val metric: {:4f}".format(best_metric))
        self.logger.info("Final test metric: {:4f}".format(final_metric))

        return self.model

    def _train_epoch(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, MetricBase] | list[MetricBase],
        device: torch.device,
    ) -> float:
        self.logger.debug("train|{}|start".format(epoch))
        self._reset_metrics(metrics, device)
        self.model.train()

        epoch_start = time.time()
        start_data_loading = epoch_start
        data_loading_time = Sum(torch.device("cpu"))
        epoch_loss = Mean(device)

        num_classes = train_loader.dataset.num_classes  # type: ignore[union-attr]
        if num_classes == 2:
            auc: AUCMeter | None = AUCMeter()
            confusion: ConfusionMeter | None = None
        else:
            auc = None
            confusion = ConfusionMeter(n_categories=num_classes)

        for i, (features, label) in enumerate(train_loader):
            features = features.to(device)
            call_label: torch.Tensor | None = None
            if "call" in label:
                call_label = label["call"].to(device, non_blocking=True, dtype=torch.int64)

            data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))

            optimizer.zero_grad()
            output = self.model(features)
            loss = loss_fn(output, call_label)
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss)

            prediction: torch.Tensor | None = None
            if call_label is not None:
                prediction = torch.argmax(output.data, dim=1)
                self._update_metrics(metrics, call_label, prediction)
                if auc is not None:
                    score = nn.functional.softmax(output, dim=1)[:, 1]
                    auc.add(score.detach(), call_label)
                if confusion is not None:
                    confusion.add(prediction, call_label)

            if i == 0:
                self._write_summaries(
                    features=features,
                    labels=call_label,
                    prediction=prediction,
                    file_names=label["file_name"],
                    epoch=epoch,
                    phase="train",
                )
            start_data_loading = time.time()

        self._write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            metrics=metrics,
            lr=optimizer.param_groups[0]["lr"],
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase="train",
        )

        if call_label is not None:
            self._write_classification_summaries(
                auc, confusion, train_loader, epoch, "train"
            )

        if self.writer is not None:
            self.writer.flush()
        return epoch_loss.get()

    def _test_epoch(
        self,
        epoch: int,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        metrics: dict[str, MetricBase] | list[MetricBase],
        device: torch.device,
        phase: str = "val",
    ) -> float:
        self.logger.debug("{}|{}|start".format(phase, epoch))
        self.model.eval()

        with torch.no_grad():
            self._reset_metrics(metrics, device)
            epoch_start = time.time()
            start_data_loading = epoch_start
            data_loading_time = Sum(torch.device("cpu"))
            epoch_loss = Mean(device)

            num_classes = test_loader.dataset.num_classes  # type: ignore[union-attr]
            if num_classes == 2:
                auc: AUCMeter | None = AUCMeter()
                confusion: ConfusionMeter | None = None
            else:
                auc = None
                confusion = ConfusionMeter(n_categories=num_classes)

            for i, (features, label) in enumerate(test_loader):
                features = features.to(device)
                call_label: torch.Tensor | None = None
                if "call" in label:
                    call_label = label["call"].to(
                        device, non_blocking=True, dtype=torch.int64
                    )

                data_loading_time.update(
                    torch.Tensor([(time.time() - start_data_loading)])
                )

                output = self.model(features)
                loss = loss_fn(output, call_label)
                epoch_loss.update(loss)

                prediction: torch.Tensor | None = None
                if call_label is not None:
                    prediction = torch.argmax(output.data, dim=1)
                    self._update_metrics(metrics, call_label, prediction)
                    if auc is not None:
                        score = nn.functional.softmax(output, dim=1)[:, 1]
                        auc.add(score, call_label)
                    if confusion is not None:
                        confusion.add(prediction, call_label)

                if i == 0:
                    self._write_summaries(
                        features=features,
                        labels=call_label,
                        prediction=prediction,
                        file_names=label["file_name"],
                        epoch=epoch,
                        phase=phase,
                    )
                start_data_loading = time.time()

        self._write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            metrics=metrics,
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase=phase,
        )

        if call_label is not None:
            self._write_classification_summaries(
                auc, confusion, test_loader, epoch, phase
            )

        if self.writer is not None:
            self.writer.flush()
        return epoch_loss.get()

    def _reset_metrics(
        self,
        metrics: dict[str, MetricBase] | list[MetricBase],
        device: torch.device,
    ) -> None:
        if isinstance(metrics, list):
            for metric in metrics:
                metric.reset(device)
        else:
            for metric in metrics.values():
                metric.reset(device)

    def _update_metrics(
        self,
        metrics: dict[str, MetricBase] | list[MetricBase],
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        if isinstance(metrics, list):
            for metric in metrics:
                metric.update(labels, predictions)
        else:
            for metric in metrics.values():
                metric.update(labels, predictions)

    def _write_summaries(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        prediction: torch.Tensor | None = None,
        file_names: list[str] | None = None,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        if self.writer is None:
            return
        with torch.no_grad():
            if file_names is not None:
                if isinstance(file_names, torch.Tensor):
                    file_names = file_names.cpu().numpy()
                elif isinstance(file_names, list):
                    file_names = np.asarray(file_names)

            if labels is not None and prediction is not None:
                features = features.cpu()
                labels = labels.cpu()
                prediction = prediction.cpu()
                matches = torch.eq(prediction, labels)
                for idx in range(len(matches)):
                    if matches[idx]:
                        name_tag = "true - {} as {}".format(
                            self._get_class_name_from_index(labels[idx].item()),
                            self._get_class_name_from_index(prediction[idx].item()),
                        )
                    else:
                        name_tag = "false - {} as {}".format(
                            self._get_class_name_from_index(labels[idx].item()),
                            self._get_class_name_from_index(prediction[idx].item()),
                        )
                    try:
                        self.writer.add_image(
                            tag=phase + "/" + name_tag,
                            img_tensor=prepare_img(
                                features[idx].unsqueeze(dim=0),
                                num_images=self.n_summaries,
                                file_names=[file_names[idx]] if file_names is not None else None,
                            ),
                            global_step=epoch,
                        )
                    except ValueError:
                        pass
            else:
                self.writer.add_image(
                    tag=phase + "/input",
                    img_tensor=prepare_img(
                        features, num_images=self.n_summaries, file_names=file_names
                    ),
                    global_step=epoch,
                )

    def _write_scalar_summaries_logs(
        self,
        loss: float,
        metrics: dict[str, MetricBase] | list[MetricBase] = {},
        lr: float | None = None,
        epoch_time: float | None = None,
        data_loading_time: float | None = None,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        with torch.no_grad():
            log_str = phase
            if epoch is not None:
                log_str += "|{}".format(epoch)

            if self.writer is not None:
                self.writer.add_scalar(phase + "/epoch_loss", loss, epoch)
            log_str += "|loss:{:0.3f}".format(loss)

            if isinstance(metrics, dict):
                for name, metric in metrics.items():
                    if self.writer is not None:
                        self.writer.add_scalar(
                            phase + "/" + name, metric.get(), epoch
                        )
                    log_str += "|{}:{:0.3f}".format(name, metric.get())
            else:
                for i, metric in enumerate(metrics):
                    if self.writer is not None:
                        self.writer.add_scalar(
                            phase + "/metric_" + str(i), metric.get(), epoch
                        )
                    log_str += "|m_{}:{:0.3f}".format(i, metric.get())

            if lr is not None:
                if self.writer is not None:
                    self.writer.add_scalar("lr", lr, epoch)
                log_str += "|lr:{:0.2e}".format(lr)

            if epoch_time is not None:
                if self.writer is not None:
                    self.writer.add_scalar(phase + "/time", epoch_time, epoch)
                log_str += "|t:{:0.1f}".format(epoch_time)

            if data_loading_time is not None and self.writer is not None:
                self.writer.add_scalar(
                    phase + "/data_loading_time", data_loading_time, epoch
                )

            self.logger.info(log_str)

    def _write_classification_summaries(
        self,
        auc: AUCMeter | None,
        confusion: ConfusionMeter | None,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        phase: str,
    ) -> None:
        if self.writer is None:
            return
        with torch.no_grad():
            if auc is not None:
                auc_val, tpr, fpr = auc.value()
                phase_prefix = phase + "_" if phase else ""
                fig = roc_fig(tpr, fpr, auc_val)
                self.writer.add_figure(phase_prefix + "roc/roc", fig, epoch)

            if confusion is not None:
                confusion_matrix_raw = confusion.confusion.clone()
                confusion_matrix_norm = confusion.value()
                label_str = [
                    loader.dataset.get_class_name_from_index(i)  # type: ignore[union-attr]
                    for i in range(confusion_matrix_norm.shape[0])
                ]
                phase_prefix = phase + "_" if phase else ""

                fig_norm = confusion_matrix_fig(
                    confusion_matrix_norm, label_str=label_str, numbering=True
                )
                self.writer.add_figure(
                    phase_prefix + "confusion_matrix_norm/cm_numbered",
                    fig_norm,
                    epoch,
                )

                fig_norm_no_num = confusion_matrix_fig(
                    confusion_matrix_norm, label_str=label_str, numbering=False
                )
                self.writer.add_figure(
                    phase_prefix + "confusion_matrix_norm/cm",
                    fig_norm_no_num,
                    epoch,
                )

                fig_raw = confusion_matrix_fig(
                    confusion_matrix_raw, label_str=label_str, numbering=True
                )
                self.writer.add_figure(
                    phase_prefix + "confusion_matrix_raw/cm_numbered",
                    fig_raw,
                    epoch,
                )

    def _get_class_name_from_index(self, idx: int) -> str:
        if self.class_dist_dict is None:
            return str(idx)
        for name, index in self.class_dist_dict.items():
            if index == idx:
                return name
        raise ValueError("Unknown class type for index {}".format(idx))
