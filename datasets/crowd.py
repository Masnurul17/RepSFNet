"""Crowd counting dataset.

Expects the *preprocessed* layout produced by ``preprocess/prepare_*.py``:

    <data_root>/
        train/  img_0001.jpg  img_0001.npy   # npy = float32 array (N, 2) of (x, y)
        val/    ...
        test/   ...

Training returns a random crop plus a ground-truth density map downsampled by
``downsample_ratio`` (the model's output stride).  The density map preserves
mass: summing it gives the number of annotated heads inside the crop.
"""

from __future__ import annotations

import os
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _points_to_density(points, height, width, ratio, sigma=0.0):
    """Rasterize points onto an (H/ratio, W/ratio) grid, preserving total mass."""
    h, w = height // ratio, width // ratio
    density = np.zeros((h, w), dtype=np.float32)
    if len(points):
        xs = np.clip((points[:, 0] / ratio).astype(np.int64), 0, w - 1)
        ys = np.clip((points[:, 1] / ratio).astype(np.int64), 0, h - 1)
        np.add.at(density, (ys, xs), 1.0)
    if sigma and sigma > 0 and density.sum() > 0:
        try:
            from scipy.ndimage import gaussian_filter

            total = density.sum()
            density = gaussian_filter(density, sigma, mode="constant")
            # re-normalize: gaussian_filter with mode='constant' loses border mass
            if density.sum() > 0:
                density *= total / density.sum()
        except ImportError:  # scipy is optional
            pass
    return density


class CrowdDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 512,
        downsample_ratio: int = 8,
        method: str = "train",
        sigma: float = 0.0,
        max_size: int = 2048,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.root = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(self.root, "*.jpg")))
        if not self.files:
            raise FileNotFoundError(
                f"no .jpg found in {self.root!r} -- did you run preprocess/prepare_*.py?"
            )
        assert method in ("train", "val")
        assert crop_size % downsample_ratio == 0, "crop_size must be divisible by the output stride"
        self.crop_size = crop_size
        self.ratio = downsample_ratio
        self.method = method
        self.sigma = sigma
        self.max_size = max_size
        self.normalize = T.Normalize(mean=mean, std=std)
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------------------ #
    def _load(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert("RGB")
        pts_path = os.path.splitext(img_path)[0] + ".npy"
        points = np.load(pts_path).astype(np.float32) if os.path.exists(pts_path) else np.zeros((0, 2), np.float32)
        return img, points, img_path

    def __getitem__(self, index):
        img, points, path = self._load(index)
        if self.method == "train":
            return self._train_item(img, points)
        return self._val_item(img, points, path)

    # ------------------------------------------------------------------ #
    def _train_item(self, img, points):
        w, h = img.size
        cs = self.crop_size

        # pad if the image is smaller than the crop
        if w < cs or h < cs:
            pad_w, pad_h = max(0, cs - w), max(0, cs - h)
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
            w, h = img.size

        i = random.randint(0, h - cs)
        j = random.randint(0, w - cs)
        img = TF.crop(img, i, j, cs, cs)

        if len(points):
            keep = (
                (points[:, 0] >= j) & (points[:, 0] < j + cs)
                & (points[:, 1] >= i) & (points[:, 1] < i + cs)
            )
            points = points[keep] - np.array([j, i], dtype=np.float32)
        else:
            points = points.reshape(0, 2)

        if random.random() < 0.5:
            img = TF.hflip(img)
            if len(points):
                points[:, 0] = cs - 1 - points[:, 0]

        if random.random() < 0.3:
            img = self.color_jitter(img)

        density = _points_to_density(points, cs, cs, self.ratio, self.sigma)
        tensor = self.normalize(TF.to_tensor(img))
        return tensor, torch.from_numpy(density).unsqueeze(0), torch.tensor(float(len(points)))

    def _val_item(self, img, points, path):
        w, h = img.size
        scale = min(1.0, self.max_size / max(w, h))
        if scale < 1.0:
            w, h = int(w * scale), int(h * scale)
            img = img.resize((w, h), Image.BILINEAR)
            points = points * scale if len(points) else points

        # make the size divisible by the network's total stride (32)
        nw, nh = (w // 32) * 32, (h // 32) * 32
        nw, nh = max(nw, 32), max(nh, 32)
        if (nw, nh) != (w, h):
            sx, sy = nw / w, nh / h
            img = img.resize((nw, nh), Image.BILINEAR)
            if len(points):
                points = points * np.array([sx, sy], dtype=np.float32)

        tensor = self.normalize(TF.to_tensor(img))
        return tensor, torch.tensor(float(len(points))), os.path.basename(path)


def train_collate(batch):
    images, densities, counts = zip(*batch)
    return torch.stack(images, 0), torch.stack(densities, 0), torch.stack(counts, 0)
