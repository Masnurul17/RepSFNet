"""Shared helpers for dataset preparation."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def resize_and_save(img: Image.Image, points: np.ndarray, out_img: str,
                    min_size: int = 512, max_size: int = 2048, quality: int = 95):
    """Rescale so that ``min_size <= min(h, w)`` and ``max(h, w) <= max_size``.

    Point coordinates are rescaled by the same factor.  The image is written as
    JPEG and the points as a float32 ``(N, 2)`` ``.npy`` next to it.
    """
    w, h = img.size
    scale = 1.0
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
    if max(w, h) * scale > max_size:
        scale = max_size / max(w, h)

    if abs(scale - 1.0) > 1e-3:
        w, h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((w, h), Image.BILINEAR)
        if len(points):
            points = points * scale

    if len(points):
        points = points[(points[:, 0] >= 0) & (points[:, 0] < w)
                        & (points[:, 1] >= 0) & (points[:, 1] < h)]
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    img.convert("RGB").save(out_img, quality=quality)
    np.save(os.path.splitext(out_img)[0] + ".npy", points)
    return len(points)


def load_mat_points(path: str, keys=("annPoints", "image_info", "annpoints")):
    """Read point annotations from a MATLAB .mat file."""
    from scipy.io import loadmat

    mat = loadmat(path)
    for key in keys:
        if key not in mat:
            continue
        value = mat[key]
        if key == "image_info":  # ShanghaiTech nesting
            return np.asarray(value[0][0][0][0][0], dtype=np.float32)
        return np.asarray(value, dtype=np.float32)
    raise KeyError(f"no known annotation key in {path} (found {list(mat.keys())})")
