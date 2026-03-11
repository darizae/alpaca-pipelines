"""
Logger factories for training and prediction.

Replaces the ANIMAL-SPOT Singleton ``Logger`` with factory functions that
return standard ``logging.Logger`` instances. Consumers configure the root
logger however they want; these factories just attach the right handlers and
formatters.

Ported from ANIMAL-SPOT ``utils/logging.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
from pathlib import Path
from typing import Optional


class TrainingLogger(logging.Logger):
    """Logger subclass with a convenience ``epoch()`` method for training output."""

    def epoch(
        self,
        phase: str,
        epoch: int,
        num_epochs: int,
        loss: float,
        accuracy: float | None = None,
        f1_score: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        lr: float | None = None,
        epoch_time: float | None = None,
    ) -> None:
        message = "{}|{:03d}/{:d}|loss:{:0.3f}".format(
            phase.upper().rjust(5, " "), epoch, num_epochs, loss
        )
        if accuracy is not None:
            message += "|acc:{:0.3f}".format(accuracy)
        if f1_score is not None:
            message += "|f1:{:0.3f}".format(f1_score)
        if precision is not None:
            message += "|pr:{:0.3f}".format(precision)
        if recall is not None:
            message += "|re:{:0.3f}".format(recall)
        if lr is not None:
            message += "|lr:{:0.2e}".format(lr)
        if epoch_time is not None:
            message += "|t:{:0.1f}".format(epoch_time)
        self.info(message)


def _build_formatter(include_name: bool = False) -> logging.Formatter:
    fmt = "%(asctime)s"
    if include_name:
        fmt += "|%(name)s"
    fmt += "|%(levelname).1s|%(message)s"
    return logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")


def create_logger(
    name: str,
    debug: bool = False,
    log_dir: Optional[str | Path] = None,
    include_name: bool = False,
) -> TrainingLogger:
    """Create a logger with stream and optional file handlers.

    Uses a ``QueueHandler`` / ``QueueListener`` pair so that logging I/O
    does not block the training loop.
    """
    logging.setLoggerClass(TrainingLogger)
    logger: TrainingLogger = logging.getLogger(name)  # type: ignore[assignment]
    logging.setLoggerClass(logging.Logger)

    if logger.handlers:
        return logger

    level = logging.DEBUG if debug else logging.INFO
    formatter = _build_formatter(include_name)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [stream_handler]

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / (name + ".log"), mode="w")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(1000)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(log_queue, *handlers)

    logger.setLevel(level)
    logger.addHandler(queue_handler)
    logger.propagate = False
    listener.start()

    logger._listener = listener  # type: ignore[attr-defined]
    logger._queue_handler = queue_handler  # type: ignore[attr-defined]

    return logger


def create_prediction_logger(
    name: str,
    debug: bool = False,
    log_dir: Optional[str | Path] = None,
    include_name: bool = False,
) -> TrainingLogger:
    """Create a non-shared prediction logger (new logger per file).

    Unlike ``create_logger``, this always creates a fresh logger even if
    one with the same name already exists, making it suitable for per-file
    prediction logging.
    """
    unique_name = f"prediction.{name}.{id(name)}"
    return create_logger(unique_name, debug=debug, log_dir=log_dir, include_name=include_name)


def close_logger(logger: logging.Logger) -> None:
    """Stop the queue listener and close handlers."""
    listener = getattr(logger, "_listener", None)
    if listener is not None:
        listener.stop()
    queue_handler = getattr(logger, "_queue_handler", None)
    if queue_handler is not None:
        queue_handler.close()
