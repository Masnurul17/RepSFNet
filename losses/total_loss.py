"""Total training objective (Eq. 3 and 4 of the paper).

    TL = MAE(count) + L_OT(z, z_hat)
       = (1/N) * sum_i |y_hat_i - y_i| + W( z/||z||_1 , z_hat/||z_hat||_1 )

MAE keeps the *global count* honest, OT keeps the *spatial distribution* honest.
An optional pixel-wise MSE term is available (``mse_weight``) because the
introduction of the paper also mentions an MSE-based density term; it is
disabled by default so that the default configuration matches Eq. 3/4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_loss import OTLoss


class RepSFNetLoss(nn.Module):
    def __init__(
        self,
        ot_weight: float = 1.0,
        mae_weight: float = 1.0,
        mse_weight: float = 0.0,
        ot_eps: float = 0.05,
        ot_iter: int = 100,
        ot_downsample: int = 1,
        count_scale: float = 1.0,
    ):
        super().__init__()
        self.ot = OTLoss(eps=ot_eps, max_iter=ot_iter, downsample=ot_downsample)
        self.ot_weight = ot_weight
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.count_scale = count_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """pred / target: B x 1 x H x W. Returns (total, log_dict)."""
        pred_count = pred.sum(dim=(1, 2, 3))
        gt_count = target.sum(dim=(1, 2, 3))

        mae = torch.abs(pred_count - gt_count).mean() * self.count_scale
        total = self.mae_weight * mae
        logs = {"mae": mae.detach()}

        if self.ot_weight > 0:
            ot = self.ot(pred, target)
            total = total + self.ot_weight * ot
            logs["ot"] = ot.detach()

        if self.mse_weight > 0:
            mse = F.mse_loss(pred, target)
            total = total + self.mse_weight * mse
            logs["mse"] = mse.detach()

        logs["total"] = total.detach()
        return total, logs
