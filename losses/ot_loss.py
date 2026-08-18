"""Optimal Transport loss (Eq. 2 of the paper).

    L_OT(z, z_hat) = W( z / ||z||_1 , z_hat / ||z_hat||_1 )

Both the ground-truth density map ``z`` and the predicted density map
``z_hat`` are normalized to unit mass and treated as probability distributions
over the pixel grid.  The Wasserstein distance ``W`` is approximated with the
entropy-regularized Sinkhorn algorithm, computed in the log domain for
numerical stability.

Unlike MAE, which only sees the total count, the OT term is sensitive to *where*
the mass sits, so it penalises density maps that get the count right but the
spatial distribution wrong.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _grid_cost(h: int, w: int, device, dtype, normalize: bool = True):
    """Squared Euclidean cost between every pair of grid cells."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # N x 2
    if normalize:
        coords = coords / max(h, w)
    cost = torch.cdist(coords, coords, p=2) ** 2
    return cost


class SinkhornDistance(nn.Module):
    """Entropy-regularized Wasserstein distance (log-domain Sinkhorn)."""

    def __init__(self, eps: float = 0.05, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor):
        """a: B x N, b: B x N (unit mass each), cost: N x N."""
        log_a = torch.log(a.clamp_min(1e-12))
        log_b = torch.log(b.clamp_min(1e-12))
        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)
        C = cost.unsqueeze(0)  # 1 x N x N

        for _ in range(self.max_iter):
            f_prev = f
            # f_i = eps * ( log a_i - logsumexp_j( (g_j - C_ij)/eps ) )
            M = (g.unsqueeze(1) - C) / self.eps
            f = self.eps * (log_a - torch.logsumexp(M, dim=2))
            M = (f.unsqueeze(2) - C) / self.eps
            g = self.eps * (log_b - torch.logsumexp(M, dim=1))
            if (f - f_prev).abs().max() < self.tol:
                break

        log_pi = (f.unsqueeze(2) + g.unsqueeze(1) - C) / self.eps
        pi = torch.exp(log_pi)
        return (pi * C).sum(dim=(1, 2))


class OTLoss(nn.Module):
    """OT loss between normalized predicted and ground-truth density maps.

    Args:
        eps:        Sinkhorn entropic regularization.
        max_iter:   Sinkhorn iterations.
        downsample: extra average-pooling factor applied to both maps before
                    computing the transport plan.  The cost matrix is
                    ``(H*W) x (H*W)``, so pooling keeps memory bounded on large
                    crops (e.g. a 512x512 crop at stride 8 gives 64x64 = 4096
                    cells -> a 4096x4096 matrix; ``downsample=2`` makes it
                    1024x1024).
    """

    def __init__(self, eps: float = 0.05, max_iter: int = 100, downsample: int = 1,
                 normalize_cost: bool = True):
        super().__init__()
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter)
        self.downsample = downsample
        self.normalize_cost = normalize_cost
        self._cost_cache: dict = {}

    def _cost(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cost_cache:
            self._cost_cache[key] = _grid_cost(h, w, device, dtype, self.normalize_cost)
        return self._cost_cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """pred / target: B x 1 x H x W (non-negative density maps)."""
        if self.downsample > 1:
            k = self.downsample
            pred = torch.nn.functional.avg_pool2d(pred, k) * (k * k)
            target = torch.nn.functional.avg_pool2d(target, k) * (k * k)

        b, _, h, w = pred.shape
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)

        # skip images with no annotated points
        mass_t = t.sum(dim=1)
        valid = mass_t > 1e-8
        if not valid.any():
            return pred.sum() * 0.0

        p = p.clamp_min(0) + 1e-8
        p = p / p.sum(dim=1, keepdim=True)
        t = t.clamp_min(0) + 1e-8
        t = t / t.sum(dim=1, keepdim=True)

        cost = self._cost(h, w, pred.device, pred.dtype)
        dist = self.sinkhorn(t, p, cost)
        return dist[valid].mean()
