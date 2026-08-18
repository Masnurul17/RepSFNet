"""Prepare NWPU-Crowd.

Expected input layout:

    <src>/images/0001.jpg
    <src>/mats/0001.mat            (variable `annPoints`)   -- or jsons/0001.json
    <src>/train.txt  <src>/val.txt  <src>/test.txt

The official test split has no public annotations; images are still exported so
that predictions can be submitted to the benchmark server.

Usage
-----
    python preprocess/prepare_nwpu.py --src /data/NWPU-Crowd --dst data/nwpu
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import load_mat_points, resize_and_save

Image.MAX_IMAGE_PIXELS = None


def read_split(path):
    if not os.path.exists(path):
        return []
    ids = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(line.split()[0])
    return ids


def load_points(src, img_id):
    mat = os.path.join(src, "mats", f"{img_id}.mat")
    if os.path.exists(mat):
        return load_mat_points(mat)
    js = os.path.join(src, "jsons", f"{img_id}.json")
    if os.path.exists(js):
        with open(js) as fh:
            data = json.load(fh)
        return np.asarray(data.get("points", []), dtype=np.float32).reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)


def convert(src, ids, split_dir, min_size, max_size):
    total = 0
    for k, img_id in enumerate(ids, 1):
        img_path = os.path.join(src, "images", f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"  missing {img_path}, skipped")
            continue
        points = load_points(src, img_id)
        total += resize_and_save(Image.open(img_path), points,
                                 os.path.join(split_dir, f"{img_id}.jpg"),
                                 min_size, max_size)
        if k % 100 == 0:
            print(f"  {k}/{len(ids)}")
    print(f"{split_dir}: {len(ids)} images, {total} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--max-size", type=int, default=2048)
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        ids = read_split(os.path.join(args.src, f"{split}.txt"))
        if not ids:
            print(f"{split}.txt not found or empty, skipped")
            continue
        convert(args.src, ids, os.path.join(args.dst, split), args.min_size, args.max_size)


if __name__ == "__main__":
    main()
