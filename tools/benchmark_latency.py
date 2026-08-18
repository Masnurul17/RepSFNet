"""Reproduce Table 6: parameters, MACs and inference latency.

The paper reports latency on an NVIDIA RTX 4070 Ti Super at 640x480, 1280x960
and 1600x1200.

    python tools/benchmark_latency.py --config configs/shanghaitech_a.yaml --deploy
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model  # noqa: E402
from utils import count_parameters, load_checkpoint, load_config  # noqa: E402

RESOLUTIONS = [(480, 640), (960, 1280), (1200, 1600)]


def macs_of(model, shape):
    try:
        from thop import profile

        x = torch.randn(1, 3, *shape)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs
    except Exception:
        return None


@torch.no_grad()
def benchmark(model, shape, device, warmup=10, iters=50):
    x = torch.randn(1, 3, *shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shanghaitech_a.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deploy", action="store_true", help="benchmark the reparameterized model")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.eval()

    params_train = count_parameters(model)
    macs = macs_of(model, (480, 640))
    if args.deploy:
        model.switch_to_deploy()
    device = torch.device(args.device)
    model.to(device)

    print(f"device            : {device}")
    print(f"mode              : {'deploy (reparameterized)' if args.deploy else 'train'}")
    print(f"parameters        : {params_train / 1e6:.2f} M "
          f"-> {count_parameters(model) / 1e6:.2f} M")
    if macs:
        print(f"MACs @ 640x480    : {macs / 1e9:.2f} G")
    else:
        print("MACs              : install `thop` to report MACs")

    for h, w in RESOLUTIONS:
        ms = benchmark(model, (h, w), device, iters=args.iters)
        print(f"latency {w}x{h:<5}: {ms:8.3f} ms  ({1000 / ms:5.1f} FPS)")


if __name__ == "__main__":
    main()
