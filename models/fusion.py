"""Feature Fusion and Concatenate Fusion modules.

Feature Fusion   : for every used backbone stage, an ASPP branch (fixed multi
                   scale context) and a CAN branch (pixel-adaptive context) are
                   computed and concatenated -- this is the "single fusion"
                   design: one fusion point per stage, no attention, no
                   multi-branch decoder.
Concatenate Fusion: the per-stage fusion outputs are resampled onto a common
                   grid, concatenated channel-wise and projected by a 1x1
                   convolution into the final density map.  Concatenation (as
                   opposed to summation) keeps semantic and spatial information
                   from every level intact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import ASPP
from .can import CAN


class UpsampleConv(nn.Module):
    """The `U` node of Figure 2: 3x3 conv + bilinear upsample (x2 each step)."""

    def __init__(self, in_channels, out_channels, steps: int = 1):
        super().__init__()
        self.steps = steps
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        for _ in range(self.steps):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x


class FeatureFusion(nn.Module):
    """ASPP + CAN on one stage, concatenated."""

    def __init__(self, in_channels: int, branch_channels: int = 256,
                 aspp_rates=(6, 12, 18, 24), can_scales=(1, 2, 3, 6),
                 reduction: int = 16):
        super().__init__()
        self.aspp = ASPP(in_channels, branch_channels, rates=aspp_rates)
        self.can = CAN(in_channels, branch_channels, scales=can_scales, reduction=reduction)
        self.out_channels = branch_channels * 2

    def forward(self, x):
        return torch.cat([self.aspp(x), self.can(x)], dim=1)


class ConcatenateFusion(nn.Module):
    """Align, concatenate and project the per-stage fusion maps."""

    def __init__(self, in_channels_list, mid_channels: int = 256):
        super().__init__()
        total = sum(in_channels_list)
        self.head = nn.Sequential(
            nn.Conv2d(total, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
            nn.ReLU(inplace=True),  # densities are non-negative
        )

    def forward(self, feats, size=None):
        if size is None:
            size = feats[0].shape[-2:]
        aligned = [
            f if f.shape[-2:] == tuple(size)
            else F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            for f in feats
        ]
        return self.head(torch.cat(aligned, dim=1))
