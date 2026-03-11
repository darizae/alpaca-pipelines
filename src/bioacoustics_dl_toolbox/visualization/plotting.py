"""
Plotting utilities for ROC curves, confusion matrices, and spectrograms.

Ported from ANIMAL-SPOT ``utils/summary.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as tick
import numpy as np
import torch

matplotlib.use("Agg")


def roc_fig(
    tpr: np.ndarray, fpr: np.ndarray, auc: float
) -> matplotlib.figure.Figure:
    """Plot an ROC curve."""
    fig = plt.figure()
    plt.plot(fpr, tpr, label="AUC: {}".format(auc))
    plt.legend(markerscale=0)
    plt.title("ROC curve")
    return fig


def confusion_matrix_fig(
    confusion: torch.Tensor | np.ndarray,
    label_str: list[str] | None = None,
    numbering: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a confusion matrix heatmap."""
    if isinstance(confusion, torch.Tensor):
        confusion = confusion.numpy()
    if label_str is None:
        label_str = [str(i) for i in range(confusion.shape[0])]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, cmap="hot_r")
    fig.colorbar(cax)

    tick_size = list(range(0, confusion.shape[0], 1))
    ax.set_xticks(np.array(tick_size))
    ax.set_xticklabels(label_str, rotation=90)
    ax.set_yticks(np.array(tick_size))
    ax.set_yticklabels(label_str)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    if numbering:
        for (i, j), z in np.ndenumerate(confusion):
            ax.text(
                j,
                i,
                "{:0.1f}".format(z),
                size="smaller",
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3"),
            )
    return fig


def plot_spectrogram(
    spectrogram: torch.Tensor | np.ndarray,
    output_filepath: str | Path | None = None,
    sr: int = 44100,
    hop_length: int = 441,
    fmin: int = 50,
    fmax: int = 12500,
    title: str = "spectrogram",
    log: bool = False,
    show: bool = True,
    axes: matplotlib.axes.Axes | None = None,
    ax_title: str | None = None,
    **kwargs: object,
) -> None:
    """Plot a spectrogram with time and frequency axes."""
    kwargs.setdefault("cmap", plt.cm.get_cmap("viridis"))
    kwargs.setdefault("rasterized", True)

    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().cpu().numpy()
    spectrogram = spectrogram.T

    figsize: Tuple[int, int] = (5, 10)
    figure = plt.figure(figsize=figsize)
    figure.suptitle(title)

    if log:
        f = np.logspace(np.log2(fmin), np.log2(fmax), num=spectrogram.shape[0], base=2)
    else:
        f = np.linspace(fmin, fmax, num=spectrogram.shape[0])
    t = np.arange(0, spectrogram.shape[1]) * hop_length / sr

    if axes is None:
        axes = plt.gca()
    if ax_title is not None:
        axes.set_title(ax_title)

    img = axes.pcolormesh(t, f, spectrogram, shading="auto", **kwargs)
    figure.colorbar(img, ax=axes)
    axes.set_xlim(t[0], t[-1])
    axes.set_ylim(f[0], f[-1])

    if log:
        axes.set_yscale("symlog", base=2)

    yaxis = axes.yaxis
    yaxis.set_major_formatter(tick.ScalarFormatter())
    xaxis = axes.xaxis
    xaxis.set_label_text("time [s]")

    if show:
        plt.show()
    if output_filepath is not None:
        plt.savefig(str(output_filepath))
    plt.close("all")
