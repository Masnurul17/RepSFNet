"""Small utilities: seeding, meters, logging, checkpointing, config loading."""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def get_logger(save_dir: str, name: str = "repsfnet"):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(path: str) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict, other: dict) -> dict:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def save_checkpoint(state: dict, save_dir: str, filename: str = "last.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(model, path: str, device="cpu", strict: bool = True):
    """Load weights, transparently handling reparameterized (deploy) checkpoints.

    A deploy checkpoint stores ``lkb_reparam.*`` tensors instead of the
    ``lkb_origin.*`` / ``small_conv.*`` pair, so the model has to be switched to
    deploy mode *before* the state dict is applied.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    is_deploy = any("lkb_reparam" in k for k in state) or bool(ckpt.get("deploy", False))
    if is_deploy and hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()

    model.load_state_dict(state, strict=strict)
    return ckpt, is_deploy


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
