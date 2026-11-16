"""A small U-Net for field / field-line segmentation.

Why a U-Net: field lines are a few pixels wide, so the decoder needs full-resolution features,
and U-Net skip connections provide exactly that. Why this small: with 16 base channels and
four down-sampling steps the network has about 2 million parameters (8 MB of weights) and
runs in about 25 ms per 640x360 image on a laptop GPU, and the task (three classes with strong colour and
shape cues) does not need more capacity to establish a baseline.

The model is a plain ``torch.nn.Module`` that maps ``(N, 3, H, W)`` normalised RGB to
``(N, classes, H, W)`` logits, with H and W multiples of ``2 ** depth``. Any network with that
contract can replace it: add it to `build_model` and name it in the training configuration.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self, num_classes: int = 3, base_channels: int = 16, depth: int = 4,
                 in_channels: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.num_classes, self.base_channels, self.depth = num_classes, base_channels, depth
        widths = [base_channels * 2 ** level for level in range(depth + 1)]
        self.stem = ConvBlock(in_channels, widths[0])
        self.down = nn.ModuleList(ConvBlock(widths[i], widths[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(ConvBlock(widths[i + 1] + widths[i], widths[i])
                                for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    @property
    def stride(self) -> int:
        """Input sides must be multiples of this."""
        return 2 ** self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for block in self.down:
            skips.append(block(F.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for block in self.up:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


MODELS = {"small_unet": SmallUNet}


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Build a model from a configuration's ``model`` section: ``{architecture: ..., **kwargs}``."""
    arguments = dict(model_cfg)
    architecture = arguments.pop("architecture", "small_unet")
    if architecture not in MODELS:
        raise ValueError(f"unknown segmentation architecture {architecture!r}; "
                         f"available: {sorted(MODELS)}")
    return MODELS[architecture](**arguments)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
