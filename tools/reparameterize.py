"""Convert a training checkpoint into its deployment (reparameterized) form.

Every ``ReparamLargeKernelConv`` folds its parallel small kernel and both BN
layers into one large depth-wise kernel with bias.  The result is numerically
equivalent but has fewer layers and no branches -- this is where RepSFNet's
latency advantage comes from.

    python tools/reparameterize.py --config configs/shanghaitech_a.yaml \
        --checkpoint runs/sha/best.pth --output runs/sha/best_deploy.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model  # noqa: E402
from utils import count_parameters, load_checkpoint, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shanghaitech_a.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--check", action="store_true", help="verify train/deploy equivalence")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    _, already = load_checkpoint(model, args.checkpoint)
    model.eval()
    if already:
        print("checkpoint is already reparameterized -- nothing to fold")
    before = count_parameters(model)

    if args.check:
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y_train = model(x)

    model.switch_to_deploy()
    after = count_parameters(model)

    if args.check:
        with torch.no_grad():
            y_deploy = model(x)
        diff = (y_train - y_deploy).abs().max().item()
        print(f"max |train - deploy| = {diff:.3e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"model": model.state_dict(), "deploy": True, "config": cfg}, args.output)
    print(f"parameters: {before / 1e6:.2f} M -> {after / 1e6:.2f} M")
    print(f"deployment checkpoint written to {args.output}")


if __name__ == "__main__":
    main()
