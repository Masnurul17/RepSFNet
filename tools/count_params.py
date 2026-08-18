"""Per-module parameter breakdown of RepSFNet.

    python tools/count_params.py --config configs/shanghaitech_a.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model  # noqa: E402
from utils import count_parameters, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shanghaitech_a.yaml")
    ap.add_argument("--input-size", type=int, nargs=2, default=[480, 640])
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg).eval()

    total = count_parameters(model)
    print(f"{'module':<28}{'params (M)':>12}{'share':>9}")
    print("-" * 49)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"{name:<28}{n / 1e6:>12.3f}{100 * n / total:>8.1f}%")
    print("-" * 49)
    print(f"{'total':<28}{total / 1e6:>12.3f}{100.0:>8.1f}%")

    model.switch_to_deploy()
    print(f"{'total (deploy)':<28}{count_parameters(model) / 1e6:>12.3f}")

    with torch.no_grad():
        y = model(torch.randn(1, 3, *args.input_size))
    print(f"\ninput {tuple(args.input_size)} -> density map {tuple(y.shape[-2:])} "
          f"(stride {args.input_size[0] // y.shape[-2]})")


if __name__ == "__main__":
    main()
