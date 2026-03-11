"""
Core evaluation metrics for classification.

Ported from ANIMAL-SPOT ``utils/metrics.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import torch


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Element-wise division, returning 0 where denominator ≤ 0."""
    result = torch.div(numerator, denominator)
    condition = torch.gt(denominator, float(0))
    return torch.where(condition, result, torch.zeros_like(result))


def _count_condition(
    condition: torch.Tensor, weights: torch.Tensor | float | None
) -> float:
    """Count weighted true elements in a boolean condition tensor."""
    with torch.no_grad():
        if weights is not None:
            if torch.is_tensor(weights):
                weights = weights.float()
            condition = torch.mul(condition.float(), weights)
        return condition.sum().item()


class MetricBase:
    """Abstract base for all metrics."""

    def __init__(self, device: torch.device | None = None) -> None:
        pass

    def reset(self, device: torch.device | None = None) -> None:
        self.__init__(device=device)  # type: ignore[misc]

    def update(self, *args: torch.Tensor, **kwargs: torch.Tensor | None) -> None:
        pass

    def _get_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    def get(self) -> float:
        return self._get_tensor().item()

    def __str__(self) -> str:
        return str(self.get())

    def __format__(self, spec: str) -> str:
        return self.get().__format__(spec)


class Sum(MetricBase):
    """Accumulates a running sum."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.value = torch.zeros(1, device=device)

    def update(
        self, values: torch.Tensor, weights: torch.Tensor | None = None
    ) -> None:
        with torch.no_grad():
            if weights is not None:
                values = torch.mul(values, weights)
            self.value += values.sum()

    def _get_tensor(self) -> torch.Tensor:
        return self.value


class Max(MetricBase):
    """Tracks the running maximum."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.value = torch.zeros(1, dtype=torch.float, device=device)

    def update(
        self, values: torch.Tensor, weights: torch.Tensor | None = None
    ) -> None:
        with torch.no_grad():
            if weights is not None:
                values = torch.mul(values, weights)
            tmp = values.max().float()
            if tmp > self.value:
                self.value = tmp

    def _get_tensor(self) -> torch.Tensor:
        return self.value


class Mean(MetricBase):
    """Tracks a running mean."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.total = torch.zeros(1, dtype=torch.float, device=device)
        self.count = torch.zeros(1, dtype=torch.float, device=device)

    def update(
        self, values: torch.Tensor, weights: torch.Tensor | float | None = None
    ) -> None:
        with torch.no_grad():
            if weights is None:
                num_values = float(values.numel())
            else:
                if torch.is_tensor(weights):
                    num_values = weights.sum().float()
                    weights = weights.float()
                else:
                    num_values = torch.mul(values.numel(), weights).float()
                values = torch.mul(values, weights)
            self.total += values.sum().float()
            self.count += num_values

    def _get_tensor(self) -> torch.Tensor:
        return _safe_div(self.total, self.count).float()


class Accuracy(MetricBase):
    """Classification accuracy."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.mean = Mean(device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_correct = torch.eq(labels, predictions).float()
            self.mean.update(is_correct, weights)

    def _get_tensor(self) -> torch.Tensor:
        return self.mean._get_tensor()


class TruePositives(MetricBase):
    """Count of true positive predictions."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.count = torch.zeros(1, dtype=torch.float, device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_true = torch.eq(labels, True)
            is_positive = torch.eq(predictions, True)
            condition = torch.mul(is_true, is_positive)
            self.count += _count_condition(condition, weights)

    def _get_tensor(self) -> torch.Tensor:
        return self.count


class FalsePositives(MetricBase):
    """Count of false positive predictions."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.count = torch.zeros(1, dtype=torch.float, device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_false = torch.eq(labels, False)
            is_positive = torch.eq(predictions, True)
            condition = torch.mul(is_false, is_positive)
            self.count += _count_condition(condition, weights)

    def _get_tensor(self) -> torch.Tensor:
        return self.count


class TrueNegatives(MetricBase):
    """Count of true negative predictions."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.count = torch.zeros(1, dtype=torch.float, device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_false = torch.eq(labels, False)
            is_negative = torch.eq(predictions, False)
            condition = torch.mul(is_false, is_negative)
            self.count += _count_condition(condition, weights)

    def _get_tensor(self) -> torch.Tensor:
        return self.count


class FalseNegatives(MetricBase):
    """Count of false negative predictions."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.count = torch.zeros(1, dtype=torch.float, device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_true = torch.eq(labels, True)
            is_negative = torch.eq(predictions, False)
            condition = torch.mul(is_true, is_negative)
            self.count += _count_condition(condition, weights)

    def _get_tensor(self) -> torch.Tensor:
        return self.count


class Precision(MetricBase):
    """Precision = TP / (TP + FP)."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.tp = TruePositives(device=device)
        self.fp = FalsePositives(device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            self.tp.update(labels, predictions, weights)
            self.fp.update(labels, predictions, weights)

    def _get_tensor(self) -> torch.Tensor:
        predicted_positive = self.tp._get_tensor() + self.fp._get_tensor()
        return torch.where(
            torch.gt(predicted_positive, 0),
            torch.div(self.tp._get_tensor(), predicted_positive),
            torch.zeros_like(predicted_positive),
        )


class Recall(MetricBase):
    """Recall (TPR) = TP / (TP + FN)."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.tp = TruePositives(device=device)
        self.fn = FalseNegatives(device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            self.tp.update(labels, predictions, weights)
            self.fn.update(labels, predictions, weights)

    def _get_tensor(self) -> torch.Tensor:
        positive = self.tp._get_tensor() + self.fn._get_tensor()
        return torch.where(
            torch.gt(positive, 0),
            torch.div(self.tp._get_tensor(), positive),
            torch.zeros_like(positive),
        )


TPR = Recall


class FPR(MetricBase):
    """False Positive Rate = FP / (FP + TN)."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.fp = FalsePositives(device=device)
        self.tn = TrueNegatives(device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            self.fp.update(labels, predictions, weights)
            self.tn.update(labels, predictions, weights)

    def _get_tensor(self) -> torch.Tensor:
        negative = self.fp._get_tensor() + self.tn._get_tensor()
        return torch.where(
            torch.gt(negative, 0),
            torch.div(self.fp._get_tensor(), negative),
            torch.zeros_like(negative),
        )


class F1Score(MetricBase):
    """F1 Score = 2 * Precision * Recall / (Precision + Recall)."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.pr = Precision(device=device)
        self.re = Recall(device=device)

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            self.pr.update(labels, predictions, weights)
            self.re.update(labels, predictions, weights)

    def _get_tensor(self) -> torch.Tensor:
        total = self.pr._get_tensor() + self.re._get_tensor()
        return torch.where(
            torch.gt(total, 0),
            torch.div(2 * self.pr._get_tensor() * self.re._get_tensor(), total),
            torch.zeros_like(total),
        )
