"""Atrous Spatial Pyramid Pooling (Figure 3 of the paper).

Parallel 3x3 dilated convolutions with rates {6, 12, 18, 24}, a 1x1 convolution
and an image-level pooling branch.  With 3x3 kernels these rates give effective
receptive fields of 13x13, 25x25, 37x37 and 49x49 without adding parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        y = self.block(x)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Global, fixed-scale context branch of the Feature Fusion module."""

    def __init__(self, in_channels: int, out_channels: int = 256,
                 rates=(6, 12, 18, 24), dropout: float = 0.1):
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        branches += [ASPPConv(in_channels, out_channels, r) for r in rates]
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.project(out)
