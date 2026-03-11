"""
Confusion matrix meter.

Based on ``torchnet`` (BSD-3-Clause, Zagoruyko & Chilamkurthy).
Ported from ANIMAL-SPOT ``utils/confusionmeter.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import numbers

import numpy as np
import torch


class ConfusionMeter:
    """Accumulates a confusion matrix over batches.

    Parameters
    ----------
    n_categories:
        Number of classes.
    """

    def __init__(self, n_categories: int) -> None:
        self.n_categories = n_categories
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(self.n_categories, self.n_categories)

    def add(
        self,
        output: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray | int,
    ) -> None:
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, "wrong output size (1D expected)"
        assert np.ndim(target) == 1, "wrong target size (1D expected)"
        assert (
            output.shape[0] == target.shape[0]
        ), "number of outputs and targets does not match"
        for output_val, target_val in zip(output, target):
            self.confusion[int(target_val)][int(output_val)] += 1

    def value(self) -> torch.Tensor:
        """Return the row-normalized confusion matrix."""
        norm_confusion = self.confusion.clone()
        for i in range(self.n_categories):
            norm_factor = norm_confusion[i].sum()
            if norm_factor == 0:
                norm_factor = 1
            norm_confusion[i] = norm_confusion[i] / norm_factor
        return norm_confusion
