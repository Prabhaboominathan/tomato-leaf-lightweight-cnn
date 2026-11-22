"""Export the trained model to ONNX for mobile / edge runtimes.

    python -m src.export_onnx --ckpt_dir results/checkpoints --out results/model.onnx
"""
import argparse
import json
from pathlib import Path

import torch

from .models import LightweightCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="results/checkpoints")
    ap.add_argument("--out", default="results/model.onnx")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--expand", type=int, default=6)
    args = ap.parse_args()
    classes = json.loads((Path(args.ckpt_dir) / "classes.json").read_text())
    model = LightweightCNN(len(classes), args.expand)
    model.load_state_dict(torch.load(Path(args.ckpt_dir) / "model.pth", map_location="cpu"))
    model.eval()
    dummy = torch.rand(1, 3, args.img_size, args.img_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, args.out, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13)
    print(f"saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
