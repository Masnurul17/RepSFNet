"""Train RepSFNet.

    python train.py --config configs/shanghaitech_a.yaml
    python train.py --config configs/nwpu.yaml --data-root data/nwpu --epochs 800

Any top-level config key can be overridden from the command line (see --help).
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from datasets import CrowdDataset, train_collate
from losses import RepSFNetLoss
from models import build_model
from utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    load_config,
    save_checkpoint,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser("RepSFNet training")
    ap.add_argument("--config", default="configs/shanghaitech_a.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--crop-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="mixed precision training")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_loaders(cfg):
    d = cfg["data"]
    train_set = CrowdDataset(
        d["root"], "train", crop_size=d["crop_size"],
        downsample_ratio=cfg["model"].get("output_stride", 8),
        method="train", sigma=d.get("sigma", 0.0),
    )
    val_split = d.get("val_split", "val")
    val_dir = os.path.join(d["root"], val_split)
    if not os.path.isdir(val_dir):
        val_split = "test"
    val_set = CrowdDataset(
        d["root"], val_split, crop_size=d["crop_size"],
        downsample_ratio=cfg["model"].get("output_stride", 8),
        method="val", max_size=d.get("max_size", 2048),
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=pin,
        drop_last=True, collate_fn=train_collate,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=cfg["train"]["num_workers"], pin_memory=pin)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = []
    for image, count, _ in loader:
        image = image.to(device, non_blocking=True)
        pred = model(image).sum().item()
        errs.append(pred - count.item())
    errs = torch.tensor(errs)
    mae = errs.abs().mean().item()
    mse = errs.pow(2).mean().sqrt().item()
    return mae, mse


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # command-line overrides
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.crop_size:
        cfg["data"]["crop_size"] = args.crop_size
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("lr", args.lr), ("num_workers", args.num_workers)):
        if value is not None:
            cfg["train"][key] = value
    save_dir = args.save_dir or cfg["train"].get("save_dir", "runs/repsfnet")

    set_seed(args.seed)
    logger = get_logger(save_dir)
    logger.info(f"config: {args.config}")
    logger.info(cfg)

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    logger.info(f"trainable parameters: {count_parameters(model) / 1e6:.2f} M")

    criterion = RepSFNetLoss(**cfg.get("loss", {})).to(device)
    train_loader, val_loader = build_loaders(cfg)

    opt_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 1e-4)
    )
    epochs = opt_cfg["epochs"]
    warmup = opt_cfg.get("warmup_epochs", 5)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_mae = 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")

    val_every = opt_cfg.get("val_every", 1)

    for epoch in range(start_epoch, epochs):
        model.train()
        meters = {"total": AverageMeter(), "mae": AverageMeter(), "ot": AverageMeter()}
        t0 = time.time()

        for image, density, _ in train_loader:
            image = image.to(device, non_blocking=True)
            density = density.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(image)
                loss, logs = criterion(pred, density)

            scaler.scale(loss).backward()
            if opt_cfg.get("clip_grad", 0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["clip_grad"])
            scaler.step(optimizer)
            scaler.update()

            for k, meter in meters.items():
                if k in logs:
                    meter.update(logs[k].item(), image.size(0))

        scheduler.step()
        logger.info(
            f"epoch {epoch + 1}/{epochs} | loss {meters['total'].avg:.3f} "
            f"| mae {meters['mae'].avg:.3f} | ot {meters['ot'].avg:.4f} "
            f"| lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.1f}s"
        )

        if (epoch + 1) % val_every == 0:
            mae, mse = evaluate(model, val_loader, device)
            logger.info(f"  val: MAE {mae:.2f}  MSE {mse:.2f}")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
                "config": cfg,
            }
            save_checkpoint(state, save_dir, "last.pth")
            if mae < best_mae:
                best_mae = mae
                state["best_mae"] = best_mae
                save_checkpoint(state, save_dir, "best.pth")
                logger.info(f"  new best MAE {best_mae:.2f} (MSE {mse:.2f})")

    logger.info(f"training finished. best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()
