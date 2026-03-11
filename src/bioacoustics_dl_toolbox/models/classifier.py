"""
Classification head for the ResNet encoder.

Ported from ANIMAL-SPOT ``models/classifier.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioacoustics_dl_toolbox.config import ClassifierConfig


class Classifier(nn.Module):
    """Fully connected classification head with global pooling.

    Parameters
    ----------
    config:
        Classifier configuration dataclass.
    """

    def __init__(self, config: ClassifierConfig = ClassifierConfig()) -> None:
        super().__init__()
        self._config = config
        self._layer_output: dict[str, torch.Tensor] = {}

        if config.pooling == "avg":
            self.pooling = lambda x: torch.mean(x, dim=-1)
        elif config.pooling == "max":
            self.pooling = lambda x: torch.max(x, dim=-1)[0]
        else:
            raise ValueError("Unknown pooling option: {}".format(config.pooling))

        self.linear = nn.Linear(config.input_channels, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), x.size(1), -1)
        hidden_layer = self.pooling(x)
        hidden_layer = hidden_layer.view(hidden_layer.size(0), -1)
        self._layer_output["hidden_layer_1"] = hidden_layer
        output_layer = self.linear(hidden_layer)
        self._layer_output["output_layer"] = output_layer
        return output_layer

    @property
    def config(self) -> ClassifierConfig:
        return self._config

    def get_layer_output(self) -> dict[str, torch.Tensor]:
        """Return intermediate layer outputs from the last forward pass."""
        return self._layer_output
