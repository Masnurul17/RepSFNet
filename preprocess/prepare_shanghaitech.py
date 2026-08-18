"""Prepare ShanghaiTech Part A / Part B.

Expected input layout (as distributed):

    <src>/part_A_final/train_data/images/IMG_1.jpg
    <src>/part_A_final/train_data/ground-truth/GT_IMG_1.mat
    <src>/part_A_final/test_data/...

Usage
-----
    python preprocess/prepare_shanghaitech.py \
        --src /data/ShanghaiTech --part A --dst data/sha --val-ratio 0.1
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


def find_part_dir(src: str, part: str) -> str:
    for name in (f"part_{part}_final", f"part_{part}", f"Part_{part}", f"part_{part.lower()}"):
        cand = os.path.join(src, name)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"cannot locate part {part} under {src}")


def convert(files, split_dir, min_size, max_size):
    total = 0
    for k, img_path in enumerate(files, 1):
        name = os.path.basename(img_path)
        gt = os.path.join(os.path.dirname(os.path.dirname(img_path)),
                          "ground-truth", "GT_" + os.path.splitext(name)[0] + ".mat")
        points = load_mat_points(gt)
        n = resize_and_save(Image.open(img_path), points,
                            os.path.join(split_dir, name), min_size, max_size)
        total += n
        if k % 50 == 0:
            print(f"  {k}/{len(files)}")
    print(f"{split_dir}: {len(files)} images, {total} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of the raw ShanghaiTech dataset")
    ap.add_argument("--part", default="A", choices=["A", "B"])
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--max-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    part_dir = find_part_dir(args.src, args.part)
    train_files = sorted(glob.glob(os.path.join(part_dir, "train_data", "images", "*.jpg")))
    test_files = sorted(glob.glob(os.path.join(part_dir, "test_data", "images", "*.jpg")))
    if not train_files:
        raise FileNotFoundError(f"no training images under {part_dir}")

    random.Random(args.seed).shuffle(train_files)
    n_val = int(len(train_files) * args.val_ratio)
    val_files, train_files = train_files[:n_val], train_files[n_val:]

    convert(train_files, os.path.join(args.dst, "train"), args.min_size, args.max_size)
    if val_files:
        convert(val_files, os.path.join(args.dst, "val"), args.min_size, args.max_size)
    convert(test_files, os.path.join(args.dst, "test"), args.min_size, args.max_size)


if __name__ == "__main__":
    main()
