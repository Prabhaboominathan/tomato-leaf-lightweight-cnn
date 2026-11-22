"""Model size and CPU latency (a rough check of edge suitability).

    python -m src.benchmark --ckpt results/checkpoints/model.pth --classes 10
"""
import argparse
import time

import torch

from .models import LightweightCNN, count_parameters


def measure_latency_ms(model, size: int = 224, runs: int = 50, warmup: int = 5, threads: int = 1) -> float:
    """Average CPU time for one image (batch size 1)."""
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - t0) / runs * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="optional checkpoint; random weights are fine for size and speed")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--expand", type=int, default=6)
    args = ap.parse_args()
    model = LightweightCNN(args.classes, args.expand)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    n = count_parameters(model)
    print(f"parameters: {n / 1e6:.2f} M | float32 size: {n * 4 / 1e6:.1f} MB | "
          f"CPU latency (1 thread): {measure_latency_ms(model, args.img_size):.1f} ms per image")


if __name__ == "__main__":
    main()
