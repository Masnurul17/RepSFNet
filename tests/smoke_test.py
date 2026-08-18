"""Smoke tests: shapes, reparameterization equivalence, loss backward, dataloader.

    python tests/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CrowdDataset, train_collate  # noqa: E402
from losses import RepSFNetLoss  # noqa: E402
from models import repsfnet  # noqa: E402
from utils import count_parameters  # noqa: E402


def test_forward_shapes():
    model = repsfnet("base").eval()
    for h, w in [(256, 256), (480, 640), (224, 320)]:
        with torch.no_grad():
            y = model(torch.randn(1, 3, h, w))
        assert y.shape == (1, 1, h // 8, w // 8), y.shape
        assert (y >= 0).all(), "density map must be non-negative"
    print(f"[ok] forward shapes           | params {count_parameters(model) / 1e6:.2f} M")


def test_variants():
    for variant in ("tiny", "base", "large"):
        m = repsfnet(variant).eval()
        with torch.no_grad():
            y = m(torch.randn(1, 3, 192, 256))
        assert y.shape == (1, 1, 24, 32), (variant, y.shape)
        print(f"[ok] variant {variant:<6}            | params {count_parameters(m) / 1e6:.2f} M")


def test_output_stride():
    for stride in (4, 8, 16):
        m = repsfnet("tiny", output_stride=stride).eval()
        with torch.no_grad():
            y = m(torch.randn(1, 3, 128, 192))
        assert y.shape[-2:] == (128 // stride, 192 // stride), (stride, y.shape)
    print("[ok] configurable output stride")


def test_reparameterization():
    """Folding the small kernel + BN into the large kernel must be exact.

    Run in float64: the fold itself is exact, but a 26 M-parameter float32
    network with randomized BN statistics accumulates enough rounding error
    over ~40 layers to hide that.
    """
    model = repsfnet("base").eval().double()
    # give BN layers non-trivial statistics so the fold is a real test
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.uniform_(0.8, 1.2)
                m.bias.normal_(0, 0.1)

    x = torch.randn(1, 3, 128, 128, dtype=torch.float64)
    with torch.no_grad():
        y_train = model(x)
    n_before = count_parameters(model)
    model.switch_to_deploy()
    with torch.no_grad():
        y_deploy = model(x)
    n_after = count_parameters(model)

    diff = (y_train - y_deploy).abs().max().item()
    rel = diff / (y_train.abs().max().item() + 1e-12)
    assert rel < 1e-8, f"reparameterization changed the output: rel {rel:.3e}"
    print(f"[ok] reparameterization       | rel err {rel:.2e} (float64) "
          f"| params {n_before / 1e6:.2f} M -> {n_after / 1e6:.2f} M")


def test_loss_backward():
    model = repsfnet("tiny")
    criterion = RepSFNetLoss(ot_weight=0.1, ot_iter=20, ot_downsample=2)
    x = torch.randn(2, 3, 128, 128)
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, 4, 5] = 3.0
    target[0, 0, 9, 2] = 1.0
    target[1, 0, 12, 12] = 7.0

    loss, logs = criterion(model(x), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
    assert torch.isfinite(loss), "non-finite loss"
    print(f"[ok] loss backward            | total {loss.item():.3f} "
          f"| mae {logs['mae'].item():.3f} | ot {logs['ot'].item():.4f}")


def test_ot_is_spatially_aware():
    """OT must penalise a correct count placed in the wrong location."""
    criterion = RepSFNetLoss(mae_weight=0.0, ot_weight=1.0, ot_iter=100)
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 2, 2] = 1.0

    near = torch.zeros_like(target)
    near[0, 0, 3, 3] = 1.0
    far = torch.zeros_like(target)
    far[0, 0, 13, 13] = 1.0

    loss_near, _ = criterion(near, target)
    loss_far, _ = criterion(far, target)
    assert loss_far > loss_near, (loss_near.item(), loss_far.item())
    print(f"[ok] OT is spatially aware    | near {loss_near.item():.4f} "
          f"< far {loss_far.item():.4f} (identical counts)")


def test_dataset_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        for split, n_img in (("train", 3), ("test", 2)):
            d = os.path.join(tmp, split)
            os.makedirs(d, exist_ok=True)
            for i in range(n_img):
                Image.fromarray(
                    (np.random.rand(600, 800, 3) * 255).astype(np.uint8)
                ).save(os.path.join(d, f"img_{i}.jpg"))
                pts = np.random.rand(40, 2).astype(np.float32) * np.array([800, 600], np.float32)
                np.save(os.path.join(d, f"img_{i}.npy"), pts)

        train_set = CrowdDataset(tmp, "train", crop_size=256, downsample_ratio=8, method="train")
        batch = train_collate([train_set[i] for i in range(3)])
        images, densities, counts = batch
        assert images.shape == (3, 3, 256, 256), images.shape
        assert densities.shape == (3, 1, 32, 32), densities.shape
        # the density map must preserve the number of points inside the crop
        assert torch.allclose(densities.sum(dim=(1, 2, 3)), counts, atol=1e-4), \
            (densities.sum(dim=(1, 2, 3)), counts)

        val_set = CrowdDataset(tmp, "test", crop_size=256, downsample_ratio=8, method="val")
        image, count, name = val_set[0]
        assert image.shape[-1] % 32 == 0 and image.shape[-2] % 32 == 0, image.shape
        assert count.item() == 40, count
        print(f"[ok] dataset roundtrip        | crop {tuple(images.shape[-2:])} "
              f"-> density {tuple(densities.shape[-2:])}, mass preserved")
    finally:
        shutil.rmtree(tmp)


def test_end_to_end_step():
    """One full optimizer step on a synthetic batch."""
    model = repsfnet("tiny")
    criterion = RepSFNetLoss(ot_weight=0.1, ot_iter=20, ot_downsample=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 3, 128, 128)
    target = torch.rand(2, 1, 16, 16) * 0.1

    before = criterion(model(x), target)[0].item()
    for _ in range(3):
        opt.zero_grad()
        loss, _ = criterion(model(x), target)
        loss.backward()
        opt.step()
    after = criterion(model(x), target)[0].item()
    print(f"[ok] optimizer step           | loss {before:.3f} -> {after:.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_forward_shapes()
    test_variants()
    test_output_stride()
    test_reparameterization()
    test_loss_backward()
    test_ot_is_spatially_aware()
    test_dataset_roundtrip()
    test_end_to_end_step()
    print("\nall smoke tests passed.")
