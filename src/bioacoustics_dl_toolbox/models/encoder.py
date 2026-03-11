"""
ResNet-based residual encoder for spectrogram feature extraction.

Ported from ANIMAL-SPOT ``models/residual_encoder.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioacoustics_dl_toolbox.config import EncoderConfig
from bioacoustics_dl_toolbox.models.blocks import (
    ResidualBase,
    get_block_sizes,
    get_block_type,
)
from bioacoustics_dl_toolbox.models.utils import get_padding


class ResidualEncoder(ResidualBase):
    """Convolutional feature extractor built from residual blocks.

    Supports ResNet-18, 34, 50, 101, and 152 architectures.

    Parameters
    ----------
    config:
        Encoder configuration dataclass.
    """

    def __init__(self, config: EncoderConfig = EncoderConfig()) -> None:
        super().__init__()
        self._config = config
        self.cur_in_ch = 64
        self.block_sizes = get_block_sizes(config.resnet_size)
        self.block_type = get_block_type(config.resnet_size)

        self.conv1 = nn.Conv2d(
            config.input_channels,
            out_channels=64,
            kernel_size=config.conv_kernel_size,
            stride=(2, 2),
            padding=get_padding(config.conv_kernel_size),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.cur_in_ch)
        self.relu1 = nn.ReLU(inplace=True)

        if config.max_pool == 1:
            self.max_pool: nn.Module | None = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=get_padding(3)
            )
            stride1: tuple[int, int] = (1, 1)
        elif config.max_pool == 0:
            self.max_pool = None
            stride1 = (2, 2)
        elif config.max_pool == 2:
            self.max_pool = None
            stride1 = (1, 1)
        else:
            raise ValueError("max_pool must be 0, 1, or 2")

        self.layer1 = self.make_layer(self.block_type, 64, self.block_sizes[0], stride1)
        self.layer2 = self.make_layer(self.block_type, 128, self.block_sizes[1], (2, 2))
        self.layer3 = self.make_layer(self.block_type, 256, self.block_sizes[2], (2, 2))
        self.layer4 = self.make_layer(self.block_type, 512, self.block_sizes[3], (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    @property
    def output_channels(self) -> int:
        """Number of output channels after the final residual stage."""
        return 512 * self.block_type.expansion

    @property
    def config(self) -> EncoderConfig:
        return self._config
