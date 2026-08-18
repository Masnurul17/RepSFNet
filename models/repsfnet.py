"""RepSFNet: A Single Fusion Network with Structural Reparameterization.

    image -> RepLK-ViT backbone -> {ASPP_i, CAN_i} per stage (Feature Fusion)
          -> Concatenate Fusion -> 1x1 conv -> density map

Counting is the integral of the predicted density map.

Note on output resolution: Figure 2 of the paper labels the density map
``H/32 x W/32``; the `U` / `U2x` nodes however resample stages 3 and 4 up onto
the stage-2 grid, so this implementation produces a stride-8 density map by
default (the usual convention for crowd counting).  Use ``output_stride`` to
change it -- the dataloader downsamples the ground truth by the same factor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .replk import RepLKViT, ReparamLargeKernelConv
from .fusion import FeatureFusion, ConcatenateFusion, UpsampleConv


class RepSFNet(nn.Module):
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
        fusion_channels: int = 96,
        head_channels: int = 256,
        aspp_rates=(6, 12, 18, 24),
        can_scales=(1, 2, 3, 6),
        reduction: int = 16,
        use_stages=(1, 2, 3),          # stage2, stage3, stage4 (0-indexed)
        output_stride: int = 8,
        deploy: bool = False,
    ):
        super().__init__()
        self.backbone = RepLKViT(
            in_channels=in_channels,
            channels=channels,
            depths=depths,
            kernel_sizes=kernel_sizes,
            small_kernel=small_kernel,
            dw_ratio=dw_ratio,
            ffn_ratio=ffn_ratio,
            drop_path_rate=drop_path_rate,
            deploy=deploy,
        )
        self.use_stages = tuple(use_stages)
        self.output_stride = output_stride

        # strides of the four backbone stages
        stage_strides = (4, 8, 16, 32)
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        fusion_out = []
        for s in self.use_stages:
            c = channels[s]
            # number of x2 upsampling steps needed to reach the output stride
            steps, stride = 0, stage_strides[s]
            while stride > output_stride:
                stride //= 2
                steps += 1
            if steps > 0:
                self.ups.append(UpsampleConv(c, fusion_channels, steps=steps))
                c = fusion_channels
            else:
                self.ups.append(nn.Identity())
            fusion = FeatureFusion(c, fusion_channels, aspp_rates, can_scales, reduction)
            self.fusions.append(fusion)
            fusion_out.append(fusion.out_channels)

        self.concat_fusion = ConcatenateFusion(fusion_out, head_channels)
        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = self.backbone(x)
        fused = []
        for i, s in enumerate(self.use_stages):
            f = self.ups[i](feats[s])
            fused.append(self.fusions[i](f))
        size = (x.shape[-2] // self.output_stride, x.shape[-1] // self.output_stride)
        return self.concat_fusion(fused, size=size)

    @torch.no_grad()
    def count(self, x):
        """Predicted head count = integral of the density map."""
        return self.forward(x).sum(dim=(1, 2, 3))

    def switch_to_deploy(self):
        """Fold every train-time branch/BN into a single large kernel."""
        for m in self.modules():
            if isinstance(m, ReparamLargeKernelConv):
                m.merge_kernel()
        return self


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def repsfnet(variant: str = "base", **kwargs) -> RepSFNet:
    cfgs = {
        # the configuration reported in the paper: 26.06 M params, ~62.6 G MACs
        "base": dict(channels=(256, 256, 512, 512), depths=(2, 2, 2, 4),
                     kernel_sizes=(13, 13, 7, 7), ffn_ratio=4.0,
                     fusion_channels=96, head_channels=256),
        # lighter variant for embedded / edge deployment
        "tiny": dict(channels=(128, 128, 256, 256), depths=(2, 2, 2, 2),
                     kernel_sizes=(13, 13, 7, 7), ffn_ratio=3.0,
                     fusion_channels=96, head_channels=128),
        # deeper stage 3, closer to the RepLKNet-31B layout
        "large": dict(channels=(256, 256, 512, 512), depths=(2, 2, 6, 4),
                      kernel_sizes=(13, 13, 13, 13), ffn_ratio=4.0,
                      fusion_channels=160, head_channels=256),
    }
    if variant not in cfgs:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(cfgs)}")
    cfg = dict(cfgs[variant])
    cfg.update(kwargs)
    return RepSFNet(**cfg)


def build_model(cfg: dict) -> RepSFNet:
    """Build from a config dict (see ``configs/*.yaml``)."""
    model_cfg = dict(cfg.get("model", {}))
    variant = model_cfg.pop("variant", "base")
    for key in ("channels", "depths", "kernel_sizes", "aspp_rates", "can_scales", "use_stages"):
        if key in model_cfg and model_cfg[key] is not None:
            model_cfg[key] = tuple(model_cfg[key])
    return repsfnet(variant, **model_cfg)
