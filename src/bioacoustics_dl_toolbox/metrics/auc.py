"""
Area Under the ROC Curve (AUC) meter.

Based on ``torchnet`` AUCMeter (BSD-3-Clause, Zagoruyko & Chilamkurthy).
Ported from ANIMAL-SPOT ``utils/aucmeter.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import numbers

import numpy as np
import torch


class AUCMeter:
    """Measures the area under the ROC curve for binary classification.

    Call ``add(output, target)`` after each batch, then ``value()`` to get
    the AUC, TPR, and FPR arrays.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.scores: np.ndarray = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets: np.ndarray = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray | int) -> None:
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
        assert np.all(
            np.add(np.equal(target, 1), np.equal(target, 0))
        ), "targets should be binary (0, 1)"
        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)

    def value(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return ``(auc, tpr, fpr)``."""
        if self.scores.shape[0] == 0:
            return (0.5, np.array([]), np.array([]))
        scores, sortind = torch.sort(
            torch.from_numpy(self.scores), dim=0, descending=True
        )
        scores = scores.numpy()
        sortind = sortind.numpy()

        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= self.targets.sum() * 1.0
        fpr /= (self.targets - 1.0).sum() * -1.0

        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0 : n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0 : n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return (area, tpr, fpr)
