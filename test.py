"""Evaluate a RepSFNet checkpoint (MAE / MSE).

    python test.py --config configs/shanghaitech_a.yaml \
                   --checkpoint runs/sha/best.pth --split test --deploy
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets import CrowdDataset
from models import build_model
from utils import count_parameters, load_checkpoint, load_config


def parse_args():
    ap = argparse.ArgumentParser("RepSFNet evaluation")
    ap.add_argument("--config", default="configs/shanghaitech_a.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deploy", action="store_true", help="reparameterize before evaluating")
    ap.add_argument("--save-csv", default=None, help="write per-image predictions here")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    device = torch.device(args.device)
    model = build_model(cfg)
    _, is_deploy = load_checkpoint(model, args.checkpoint)
    if args.deploy:
        model.switch_to_deploy()
        is_deploy = True
    model.to(device).eval()
    print(f"parameters: {count_parameters(model) / 1e6:.2f} M "
          f"({'deploy' if is_deploy else 'train'} mode)")

    dataset = CrowdDataset(
        cfg["data"]["root"], args.split,
        crop_size=cfg["data"]["crop_size"],
        downsample_ratio=cfg["model"].get("output_stride", 8),
        method="val", max_size=cfg["data"].get("max_size", 2048),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=min(4, os.cpu_count() or 1))

    rows, errors = [], []
    for image, count, name in loader:
        pred = model(image.to(device)).sum().item()
        gt = count.item()
        errors.append(pred - gt)
        rows.append((name[0], gt, pred))

    errors = torch.tensor(errors)
    mae = errors.abs().mean().item()
    mse = errors.pow(2).mean().sqrt().item()
    print(f"{args.split}: {len(rows)} images | MAE {mae:.2f} | MSE {mse:.2f}")

    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        with open(args.save_csv, "w") as fh:
            fh.write("image,gt,pred\n")
            for name, gt, pred in rows:
                fh.write(f"{name},{gt:.1f},{pred:.2f}\n")
        print(f"per-image predictions written to {args.save_csv}")


if __name__ == "__main__":
    main()
