"""Run RepSFNet on a single image and save a density-map visualization.

    python demo.py --checkpoint runs/sha/best.pth --image samples/crowd.jpg \
                   --output out.png --deploy
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import build_model
from utils import load_checkpoint, load_config


def parse_args():
    ap = argparse.ArgumentParser("RepSFNet demo")
    ap.add_argument("--config", default="configs/shanghaitech_a.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="density.png")
    ap.add_argument("--max-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deploy", action="store_true")
    return ap.parse_args()


def colorize(density: np.ndarray) -> Image.Image:
    """Jet-like colormap without a matplotlib dependency."""
    d = density - density.min()
    d = d / (d.max() + 1e-8)
    r = np.clip(1.5 - np.abs(4 * d - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * d - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * d - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint)
    if args.deploy:
        model.switch_to_deploy()
    model.to(args.device).eval()

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    scale = min(1.0, args.max_size / max(w, h))
    w, h = int(w * scale) // 32 * 32, int(h * scale) // 32 * 32
    img_resized = img.resize((max(w, 32), max(h, 32)), Image.BILINEAR)

    tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img_resized).unsqueeze(0).to(args.device)

    density = model(tensor)[0, 0].cpu().numpy()
    print(f"predicted count: {density.sum():.1f}")

    heat = colorize(density).resize(img.size, Image.BILINEAR)
    blended = Image.blend(img, heat, alpha=0.6)
    Image.fromarray(np.concatenate([np.array(img), np.array(blended)], axis=1)).save(args.output)
    print(f"visualization saved to {args.output}")


if __name__ == "__main__":
    main()
