"""Prepare UCF-QNRF.

Expected input layout:

    <src>/Train/img_0001.jpg
    <src>/Train/img_0001_ann.mat        (variable `annPoints`)
    <src>/Test/...

UCF-QNRF images are very large (up to 6000 px), so they are rescaled to at most
``--max-size`` pixels on the long side; annotations are rescaled accordingly.

Usage
-----
    python preprocess/prepare_qnrf.py --src /data/UCF-QNRF --dst data/qnrf
"""

from __future__ import annotations

import argparse
import glob
import os
import random

from PIL import Image

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import load_mat_points, resize_and_save

Image.MAX_IMAGE_PIXELS = None


def convert(files, split_dir, min_size, max_size):
    total = 0
    for k, img_path in enumerate(files, 1):
        gt = os.path.splitext(img_path)[0] + "_ann.mat"
        points = load_mat_points(gt)
        total += resize_and_save(Image.open(img_path), points,
                                 os.path.join(split_dir, os.path.basename(img_path)),
                                 min_size, max_size)
        if k % 50 == 0:
            print(f"  {k}/{len(files)}")
    print(f"{split_dir}: {len(files)} images, {total} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--max-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_files = sorted(glob.glob(os.path.join(args.src, "Train", "*.jpg")))
    test_files = sorted(glob.glob(os.path.join(args.src, "Test", "*.jpg")))
    if not train_files:
        raise FileNotFoundError(f"no images under {os.path.join(args.src, 'Train')}")

    random.Random(args.seed).shuffle(train_files)
    n_val = int(len(train_files) * args.val_ratio)
    val_files, train_files = train_files[:n_val], train_files[n_val:]

    convert(train_files, os.path.join(args.dst, "train"), args.min_size, args.max_size)
    if val_files:
        convert(val_files, os.path.join(args.dst, "val"), args.min_size, args.max_size)
    convert(test_files, os.path.join(args.dst, "test"), args.min_size, args.max_size)


if __name__ == "__main__":
    main()
