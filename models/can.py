"""Context-Aware Network (CAN) block.

ASPP contributes fixed-scale context; CAN adds *pixel-wise adaptivity*.  Scale
aware features are pooled at several block sizes, compared against the local
feature via a contrast term, and weighted per pixel by a learnt attention map
(channel reduction ratio r = 16).

Reference: Liu, Salzmann and Fua, "Context-Aware Crowd Counting", CVPR 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256,
                 scales=(1, 2, 3, 6), reduction: int = 16):
        super().__init__()
        self.scales = tuple(scales)

        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in self.scales
            ]
        )
        # per-pixel weight predicted from the contrast feature (local - context)
        hidden = max(out_channels // reduction, 8)
        self.weight_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, hidden, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, 1, 1, bias=True),
                )
                for _ in self.scales
            ]
        )
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x):
        size = x.shape[-2:]
        local = self.local(x)

        weights, contexts = [], []
        for s, conv, wnet in zip(self.scales, self.scale_convs, self.weight_nets):
            ctx = F.adaptive_avg_pool2d(x, output_size=s)
            ctx = conv(ctx)
            ctx = F.interpolate(ctx, size=size, mode="bilinear", align_corners=False)
            contexts.append(ctx)
            weights.append(wnet(local - ctx))

        w = torch.softmax(torch.cat(weights, dim=1), dim=1)  # B x S x H x W
        ctx = sum(w[:, i: i + 1] * contexts[i] for i in range(len(contexts)))
        return self.fuse(torch.cat([local, ctx], dim=1))
