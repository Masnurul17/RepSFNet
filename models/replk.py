"""RepLK-ViT backbone.

Convolutional backbone for RepSFNet built from large depth-wise kernels that are
*structurally reparameterized* at inference time.  During training every large
kernel is accompanied by a small parallel kernel (each followed by its own BN);
at deployment the small kernel and both BN statistics are folded into a single
large kernel, so the deployed network is a plain feed-forward CNN with no extra
branches and no attention.

Reference: Ding et al., "Scaling Up Your Kernels to 31x31: Revisiting Large
Kernel Design in CNNs", CVPR 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def conv_bn(in_ch, out_ch, kernel_size, stride=1, padding=None, dilation=1, groups=1):
    if padding is None:
        padding = dilation * (kernel_size - 1) // 2
    m = nn.Sequential()
    m.add_module(
        "conv",
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False),
    )
    m.add_module("bn", nn.BatchNorm2d(out_ch))
    return m


def conv_bn_relu(in_ch, out_ch, kernel_size, stride=1, padding=None, dilation=1, groups=1):
    m = conv_bn(in_ch, out_ch, kernel_size, stride, padding, dilation, groups)
    m.add_module("relu", nn.ReLU(inplace=True))
    return m


def fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fold a BatchNorm into the preceding convolution."""
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class DropPath(nn.Module):
    """Stochastic depth (per sample)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"p={self.p}"


# --------------------------------------------------------------------------- #
# reparameterizable large kernel
# --------------------------------------------------------------------------- #
class ReparamLargeKernelConv(nn.Module):
    """Depth-wise large kernel with a parallel small kernel (train time only).

    train : y = BN(dw_k(x)) + BN(dw_s(x))
    deploy: y = dw_k'(x) + b          (single fused kernel, bias included)
    """

    def __init__(self, channels, kernel_size, stride=1, small_kernel=3, deploy=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.stride = stride
        padding = kernel_size // 2

        if deploy:
            self.lkb_reparam = nn.Conv2d(
                channels, channels, kernel_size, stride, padding,
                dilation=1, groups=channels, bias=True,
            )
        else:
            self.lkb_origin = conv_bn(
                channels, channels, kernel_size, stride, padding, groups=channels
            )
            if small_kernel is not None:
                assert small_kernel <= kernel_size, "small kernel must not exceed the large one"
                self.small_conv = conv_bn(
                    channels, channels, small_kernel, stride,
                    small_kernel // 2, groups=channels,
                )

    def forward(self, x):
        if hasattr(self, "lkb_reparam"):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out = out + self.small_conv(x)
        return out

    # -- reparameterization ------------------------------------------------- #
    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b = eq_b + small_b
            pad = (self.kernel_size - self.small_kernel) // 2
            eq_k = eq_k + F.pad(small_k, [pad] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        if hasattr(self, "lkb_reparam"):
            return
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            self.channels, self.channels, self.kernel_size, self.stride,
            self.kernel_size // 2, dilation=1, groups=self.channels, bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
class RepLKBlock(nn.Module):
    """1x1 -> large depth-wise kernel -> 1x1, with an identity shortcut."""

    def __init__(self, channels, dw_channels, kernel_size, small_kernel,
                 drop_path=0.0, deploy=False):
        super().__init__()
        self.pw1 = conv_bn_relu(channels, dw_channels, 1)
        self.pw2 = conv_bn(dw_channels, channels, 1)
        self.large_kernel = ReparamLargeKernelConv(
            dw_channels, kernel_size, stride=1, small_kernel=small_kernel, deploy=deploy
        )
        self.lk_nonlinear = nn.ReLU(inplace=True)
        self.prelkb_bn = nn.BatchNorm2d(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class ConvFFN(nn.Module):
    """Depth-wise-free inverted bottleneck (the `1x1, 192 -> 1x1, 96` path)."""

    def __init__(self, channels, internal_channels, out_channels, drop_path=0.0):
        super().__init__()
        self.preffn_bn = nn.BatchNorm2d(channels)
        self.pw1 = conv_bn(channels, internal_channels, 1)
        self.pw2 = conv_bn(internal_channels, out_channels, 1)
        self.nonlinear = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKStage(nn.Module):
    def __init__(self, channels, depth, kernel_size, small_kernel,
                 dw_ratio=1.0, ffn_ratio=4.0, drop_path=0.0, deploy=False):
        super().__init__()
        dp = drop_path if isinstance(drop_path, (list, tuple)) else [drop_path] * depth
        blocks = []
        for i in range(depth):
            blocks.append(
                RepLKBlock(channels, int(channels * dw_ratio), kernel_size,
                           small_kernel, dp[i], deploy)
            )
            blocks.append(
                ConvFFN(channels, int(channels * ffn_ratio), channels, dp[i])
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# --------------------------------------------------------------------------- #
# backbone
# --------------------------------------------------------------------------- #
class RepLKViT(nn.Module):
    """Hierarchical large-kernel backbone (`RepLK-ViT` in the paper).

    stem (s4) -> stage1 (H/4) -> s2 -> stage2 (H/8) -> s2 -> stage3 (H/16)
              -> s2 -> stage4 (H/32)

    ``forward`` returns the four stage outputs; RepSFNet consumes stages 2-4.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels=(256, 256, 512, 512),
        depths=(2, 2, 2, 2),
        kernel_sizes=(13, 13, 7, 7),
        small_kernel: int = 3,
        dw_ratio: float = 1.0,
        ffn_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        deploy: bool = False,
    ):
        super().__init__()
        assert len(channels) == len(depths) == len(kernel_sizes) == 4
        self.channels = tuple(channels)
        self.deploy = deploy

        # 4x4 convolutional stem, stride 4
        self.stem = conv_bn_relu(in_channels, channels[0], kernel_size=4, stride=4, padding=0)

        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total)] if total else []

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(
                RepLKStage(
                    channels[i], depths[i], kernel_sizes[i], small_kernel,
                    dw_ratio, ffn_ratio, dpr[cur:cur + depths[i]], deploy,
                )
            )
            cur += depths[i]
            if i < 3:  # 3x3 stride-2 transition
                self.transitions.append(conv_bn_relu(channels[i], channels[i + 1], 3, stride=2))

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < 3:
                x = self.transitions[i](x)
        return feats  # [H/4, H/8, H/16, H/32]

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, ReparamLargeKernelConv):
                m.merge_kernel()
        self.deploy = True
        return self
