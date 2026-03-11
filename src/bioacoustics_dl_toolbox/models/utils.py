"""
Model utility functions.

Ported from ANIMAL-SPOT ``models/utils.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations


def get_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    """Return ``same`` padding for a given kernel size."""
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(s // 2 for s in kernel_size)
