"""
Residual block definitions and base class.

Ported from ANIMAL-SPOT ``models/residual_base.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

from typing import Type

import torch.nn as nn

from bioacoustics_dl_toolbox.models.utils import get_padding


class BasicBlock(nn.Module):
    """Residual basic block: two 3×3 convolutions with batch norm and ReLU."""

    expansion: int = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int | tuple[int, int] = 1,
        downsample: nn.Module | None = None,
        upsample: nn.Module | None = None,
        mid_ch: int | None = None,
    ) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        if downsample is not None and upsample is not None:
            raise ValueError("Either downsample or upsample has to be None")

        if upsample is None:
            self.shortcut = downsample
            self.conv1 = nn.Conv2d(
                in_ch,
                mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                bias=False,
            )
        else:
            self.shortcut = upsample
            self.conv1 = nn.ConvTranspose2d(
                in_ch,
                mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                output_padding=get_padding(stride),
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            mid_ch, out_ch, kernel_size=3, stride=1, padding=get_padding(3), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    """Residual bottleneck block: 1×1 → 3×3 → 1×1 convolutions."""

    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int | tuple[int, int] = 1,
        downsample: nn.Module | None = None,
        upsample: nn.Module | None = None,
        mid_ch: int | None = None,
    ) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        if downsample is not None and upsample is not None:
            raise ValueError("Either downsample or upsample has to be None")

        self.shortcut: nn.Module | None = None
        if upsample is not None or downsample is not None:
            assert (
                downsample is None or upsample is None
            ), "Only can downsample (encoder) or upsample (decoder) using the shortcut"
            self.shortcut = downsample if downsample is not None else upsample

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)

        if upsample is not None:
            self.conv2 = nn.ConvTranspose2d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                output_padding=get_padding(stride),
                bias=False,
            )
        else:
            self.conv2 = nn.Conv2d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                bias=False,
            )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=mid_ch,
            out_channels=out_ch * self.expansion,
            kernel_size=1,
            stride=1,
            padding=get_padding(1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu3(out)
        return out


class ResidualBase(nn.Module):
    """Base class providing ``make_layer`` for building residual stages."""

    def __init__(self) -> None:
        super().__init__()
        self.cur_in_ch = 64

    def make_layer(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        out_ch: int,
        size: int,
        stride: int | tuple[int, int] = 1,
        shortcut: str = "downsample",
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        stride_mean = stride
        if isinstance(stride, tuple):
            stride_mean = sum(stride) / len(stride)

        if shortcut == "upsample" and (
            stride_mean > 1 or self.cur_in_ch != out_ch * block.expansion
        ):
            upsample_module = nn.ConvTranspose2d(
                in_channels=self.cur_in_ch,
                out_channels=out_ch * block.expansion,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                output_padding=get_padding(stride),
                bias=False,
            )
            layers.append(
                block(
                    in_ch=self.cur_in_ch,
                    out_ch=out_ch,
                    mid_ch=self.cur_in_ch // block.expansion,
                    stride=stride,
                    upsample=upsample_module,
                )
            )
        elif shortcut == "downsample" and (
            stride_mean > 1 or self.cur_in_ch != out_ch * block.expansion
        ):
            downsample_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.cur_in_ch,
                    out_channels=out_ch * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
            layers.append(block(self.cur_in_ch, out_ch, stride, downsample=downsample_module))
        else:
            layers.append(block(self.cur_in_ch, out_ch))

        self.cur_in_ch = out_ch * block.expansion
        for _ in range(1, size):
            layers.append(block(self.cur_in_ch, out_ch))

        return nn.Sequential(*layers)


def get_block_sizes(resnet_size: int) -> list[int]:
    """Return the number of blocks per stage for the given ResNet variant."""
    sizes = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }
    if resnet_size not in sizes:
        raise ValueError("Unsupported resnet size: {}".format(resnet_size))
    return sizes[resnet_size]


def get_block_type(resnet_size: int) -> Type[BasicBlock] | Type[Bottleneck]:
    """Return the block class for the given ResNet variant."""
    if resnet_size in [18, 34]:
        return BasicBlock
    elif resnet_size in [50, 101, 152]:
        return Bottleneck
    raise ValueError("Unsupported resnet size: {}".format(resnet_size))
