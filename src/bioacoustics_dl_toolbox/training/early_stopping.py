"""
Early stopping criterion.

Based on PyTorch PR #7661 (BSD-style, Facebook et al.).
Ported from ANIMAL-SPOT ``utils/early_stopping.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

from typing import Any, Literal


class EarlyStoppingCriterion:
    """Stop training when a validation metric stops improving.

    Parameters
    ----------
    patience:
        Number of evaluations without improvement before stopping.
    mode:
        ``"min"`` if lower is better, ``"max"`` if higher is better.
    min_delta:
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int,
        mode: Literal["min", "max"],
        min_delta: float = 0.0,
    ) -> None:
        assert patience >= 0
        assert mode in {"min", "max"}
        assert min_delta >= 0.0

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._count = 0
        self._best_score: float | None = None
        self.is_improved: bool | None = None

    def step(self, cur_score: float) -> bool:
        """Record a new score. Returns ``True`` if training should stop."""
        if self._best_score is None:
            self._best_score = cur_score
            return False

        if self.mode == "max":
            self.is_improved = cur_score >= self._best_score + self.min_delta
        else:
            self.is_improved = cur_score <= self._best_score - self.min_delta

        if self.is_improved:
            self._count = 0
            self._best_score = cur_score
        else:
            self._count += 1

        return self._count > self.patience

    def state_dict(self) -> dict[str, Any]:
        return self.__dict__

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
