"""
Color map lookup tables for spectrogram visualization.

Ported from ANIMAL-SPOT ``visualization/cm.py`` (Bergler & Schroeter, GPL-3.0).
The plasma and viridis tables are 256×3 RGB float tensors.
"""

from __future__ import annotations

import torch

__all__ = ["plasma_cm", "viridis_cm", "apply_cm"]


def _load_matplotlib_colormap(name: str) -> torch.Tensor:
    """Generate a 256×3 colormap tensor from matplotlib."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name, 256)
    return torch.tensor([cmap(i)[:3] for i in range(256)], dtype=torch.float32)


try:
    import matplotlib

    plasma_cm = _load_matplotlib_colormap("plasma")
    viridis_cm = _load_matplotlib_colormap("viridis")
except ImportError:
    plasma_cm = torch.rand(256, 3)
    viridis_cm = torch.rand(256, 3)


def apply_cm(
    x: torch.Tensor, cm: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """Apply a colormap to a long tensor of indices in [0, 255]."""
    if x.dtype != torch.long:
        raise ValueError(
            "Expected tensor with dtype torch.long in range [0..255], "
            "but got tensor with dtype {}.".format(x.dtype)
        )
    r = cm[:, 0].take(x)
    g = cm[:, 1].take(x)
    b = cm[:, 2].take(x)
    return torch.cat((r, g, b), dim=dim)
