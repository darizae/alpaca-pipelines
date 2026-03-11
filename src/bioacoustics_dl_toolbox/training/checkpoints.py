"""
Checkpoint handling and model persistence.

Ported from ANIMAL-SPOT ``utils/checkpoints.py`` (Bergler & Schroeter, GPL-3.0).
Adds typed ``save_model`` / ``load_model`` helpers that serialize config
dataclasses alongside state dicts.
"""

from __future__ import annotations

import glob
import logging
import os
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bioacoustics_dl_toolbox.config import (
    ClassifierConfig,
    EncoderConfig,
    SpectrogramConfig,
)


class CheckpointHandler:
    """Async checkpoint writer with automatic rotation.

    Parameters
    ----------
    checkpoint_dir:
        Directory to write checkpoints into.
    prefix:
        Filename prefix for checkpoint files.
    max_checkpoints:
        Maximum number of checkpoints to keep on disk.
    logger:
        Optional logger instance.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str = "",
        max_checkpoints: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            return
        if not prefix.endswith("_"):
            prefix += "_"
        self.prefix = prefix
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self._logger = logger
        if self._logger is not None:
            self._logger.debug("Starting checkpoint writer thread")
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._should_stop = threading.Event()
        self._worker = threading.Thread(target=self._write_worker, daemon=True)
        self._worker.start()

    def _write_worker(self) -> None:
        while not self._should_stop.is_set():
            try:
                checkpoint_dict = self._queue.get()
                if checkpoint_dict is None:
                    break
                checkpoint_name = os.path.join(
                    self.checkpoint_dir,
                    "{}epoch_{:05d}.checkpoint".format(
                        self.prefix, checkpoint_dict["trainState"]["epoch"]
                    ),
                )
                if self._logger is not None:
                    self._logger.debug(
                        "Writing checkpoint to {}".format(checkpoint_name)
                    )
                torch.save(checkpoint_dict, checkpoint_name)
                checkpoints = glob.glob(
                    os.path.join(self.checkpoint_dir, self.prefix + "*.checkpoint")
                )
                if len(checkpoints) > self.max_checkpoints:
                    checkpoints.sort()
                    for i in range(len(checkpoints) - self.max_checkpoints):
                        os.remove(checkpoints[i])
            except Exception as e:
                if self._logger is not None:
                    self._logger.error("Failed to write checkpoint: {}".format(str(e)))
            finally:
                self._queue.task_done()
        if self._logger is not None:
            self._logger.info("Shutting down checkpoint writer thread")

    def write(self, checkpoint_dict: dict[str, Any]) -> None:
        """Enqueue a checkpoint for async writing."""
        if self.checkpoint_dir is None:
            return
        self._queue.put(checkpoint_dict)

    def read_latest(self) -> dict[str, Any] | None:
        """Read the most recent checkpoint, or ``None`` if none exists."""
        if self.checkpoint_dir is None:
            return None
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, self.prefix + "*.checkpoint")
        )
        if len(checkpoints) == 0:
            if self._logger is not None:
                self._logger.info(
                    "No checkpoints found in {}".format(self.checkpoint_dir)
                )
            return None
        checkpoints.sort()
        if self._logger is not None:
            self._logger.info("Restoring checkpoint {}".format(checkpoints[-1]))
        return torch.load(checkpoints[-1], map_location="cpu")  # type: ignore[no-any-return]

    def _shutdown_worker(self) -> None:
        self._should_stop.set()
        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join()

    def __del__(self) -> None:
        if self._logger is not None:
            self._logger.info("Shutting down CheckpointHandler")
        self._shutdown_worker()


def save_model(
    model: nn.Module,
    encoder: nn.Module,
    encoder_config: EncoderConfig,
    classifier: nn.Module,
    classifier_config: ClassifierConfig,
    spec_config: SpectrogramConfig,
    path: str | Path,
    class_dist_dict: dict[str, int],
) -> None:
    """Save a trained model with all configs needed to reload it.

    The saved dict contains:
    - ``encoderConfig``: serialized ``EncoderConfig``
    - ``classifierConfig``: serialized ``ClassifierConfig``
    - ``spectrogramConfig``: serialized ``SpectrogramConfig``
    - ``encoderState``: encoder state dict
    - ``classifierState``: classifier state dict
    - ``classes``: class name → index mapping
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu()
    encoder = encoder.cpu()
    classifier = classifier.cpu()

    save_dict = {
        "encoderConfig": asdict(encoder_config),
        "classifierConfig": asdict(classifier_config),
        "spectrogramConfig": asdict(spec_config),
        "encoderState": encoder.state_dict(),
        "classifierState": classifier.state_dict(),
        "classes": class_dist_dict,
    }
    torch.save(save_dict, str(path))


def load_model(
    path: str | Path,
) -> dict[str, Any]:
    """Load a model dict saved by ``save_model``.

    Returns the raw dict. The consumer is responsible for instantiating
    the encoder and classifier from the stored configs and loading
    state dicts.
    """
    return torch.load(str(path), map_location="cpu")  # type: ignore[no-any-return]
