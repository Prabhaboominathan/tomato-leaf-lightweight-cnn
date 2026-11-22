"""Predict the disease of one leaf image.

    python -m src.predict --image leaf.jpg --ckpt_dir results/checkpoints
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .models import LightweightCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt_dir", default="results/checkpoints")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--expand", type=int, default=6)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    classes = json.loads((Path(args.ckpt_dir) / "classes.json").read_text())
    model = LightweightCNN(len(classes), args.expand)
    model.load_state_dict(torch.load(Path(args.ckpt_dir) / "model.pth", map_location="cpu"))
    model.eval()
    img = Image.open(args.image).convert("RGB").resize((args.img_size, args.img_size), Image.BILINEAR)
    x = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    for p, i in zip(*torch.topk(probs, min(args.top, len(classes)))):
        print(f"{classes[int(i)]}: {float(p):.3f}")


if __name__ == "__main__":
    main()
