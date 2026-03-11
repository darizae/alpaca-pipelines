"""
Spectrogram-to-image conversion utilities.

Ported from ANIMAL-SPOT ``visualization/utils.py`` and ``utils/summary.py``
(Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from bioacoustics_dl_toolbox.visualization.colormaps import apply_cm, viridis_cm


def flip(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Flip a tensor along the given dimension.

    Based on https://github.com/pytorch/pytorch/issues/229.
    Modified by Bergler & Schroeter.
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


def spec2img(
    spec: torch.Tensor,
    normalize: bool = True,
    cm: torch.Tensor = viridis_cm,
) -> torch.Tensor:
    """Convert a float spectrogram tensor to a uint8 RGB image tensor."""
    with torch.no_grad():
        if spec.dim() == 4:
            dim = 1
            assert spec.size(1) == 1
        elif spec.dim() <= 3:
            dim = 0
            if spec.size(0) > 1 or spec.dim() == 2:
                spec = spec.unsqueeze(dim=0)
        else:
            raise ValueError("Unsupported spec dimension.")

        img = flip(spec, dim=-1)
        if normalize:
            img -= img.min()
            img /= img.max() + 1e-8
        img = img.mul(255).clamp(0, 255).long()
        img = apply_cm(img.cpu(), cm, dim=dim)
        return img.mul(255).clamp(0, 255).byte()


def prepare_img(
    img: torch.Tensor,
    num_images: int = 4,
    file_names: list[str] | np.ndarray | None = None,
) -> np.ndarray:
    """Prepare spectrogram images for TensorBoard visualization.

    Returns a numpy array suitable for ``SummaryWriter.add_image``.
    """
    with torch.no_grad():
        if img.shape[0] == 0:
            raise ValueError("`img` must include at least 1 image.")
        if num_images < img.shape[0]:
            tmp = img[:num_images]
        else:
            tmp = img
        tmp = spec2img(tmp)
        if file_names is not None:
            tmp = tmp.permute(0, 3, 2, 1)
            for i in range(tmp.shape[0]):
                try:
                    pil = Image.fromarray(tmp[i].numpy(), mode="RGB")
                    draw = ImageDraw.Draw(pil)
                    draw.text(
                        (2, 2),
                        os.path.basename(file_names[i]),
                        (255, 255, 255),
                    )
                    np_pil = np.asarray(pil).copy()
                    tmp[i] = torch.as_tensor(np_pil)
                except TypeError:
                    pass
            tmp = tmp.permute(0, 3, 1, 2)
        tmp = make_grid(tmp, nrow=1)
        return tmp.numpy()
