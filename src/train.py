"""Train and evaluate the lightweight SE + Swish CNN on tomato leaf images.

    python -m src.train --data_dir data/plantvillage/color --out_dir results
Ablations:
    python -m src.train --data_dir ... --out_dir results_no_se --no_se
    python -m src.train --data_dir ... --out_dir results_relu --activation relu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import reporting
from .benchmark import measure_latency_ms
from .data import list_images, split_indices
from .engine import LeafDataset, fit, predict_proba, set_seed
from .models import LightweightCNN, count_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/plantvillage/color")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--prefix", default="Tomato", help="only class folders starting with this text are used")
    ap.add_argument("--max_per_class", type=int, default=2000, help="0 = use every image")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--expand", type=int, default=6, help="bottleneck expansion ratio")
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--no_se", action="store_true", help="ablation: remove Squeeze-and-Excitation blocks")
    ap.add_argument("--activation", choices=["swish", "relu"], default="swish")
    ap.add_argument("--class_weights", action="store_true", help="weight the loss by inverse class frequency")
    ap.add_argument("--patience", type=int, default=0, help="early-stopping patience (0 = train all epochs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = list_images(args.data_dir, args.prefix, args.max_per_class, args.seed)
    k = len(class_names)
    tr, va, te = split_indices(labels, 0.15, 0.15, args.seed)
    print(f"{len(paths)} images, {k} classes: {class_names}\ntrain {len(tr)} | val {len(va)} | test {len(te)} | device {device}")

    def loader(idx, train):
        ds = LeafDataset([paths[i] for i in idx], labels[idx], train, args.img_size)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                          pin_memory=device.type == "cuda")

    model = LightweightCNN(k, args.expand, not args.no_se, args.activation, args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"model: {n_params / 1e6:.2f} M parameters")
    weights = len(tr) / (k * np.bincount(labels[tr], minlength=k)) if args.class_weights else None

    hist = fit(model, loader(tr, True), loader(va, False), args.epochs, args.lr, args.weight_decay,
               device, "LightweightCNN", weights, args.patience)
    torch.save(model.state_dict(), out / "checkpoints" / "model.pth")
    (out / "checkpoints" / "classes.json").write_text(json.dumps(class_names))

    probs = predict_proba(model, loader(te, False), device)
    y_true, y_pred = labels[te], probs.argmax(1)
    metrics = reporting.summary_metrics(y_true, y_pred, probs)
    metrics.update(reporting.extended_metrics(y_true, y_pred, probs))
    metrics.update({"parameters_millions": n_params / 1e6, "model_size_mb_fp32": n_params * 4 / 1e6,
                    "cpu_latency_ms": measure_latency_ms(model, args.img_size), "test_images": int(len(te)),
                    "epochs_run": len(hist["train_loss"])})

    per_class = reporting.save_all_for_model("model", y_true, y_pred, probs, class_names, out)
    log = reporting.history_table(hist)
    log.to_csv(out / "training_log.csv", index=False)
    reporting.plot_history(hist, out / "training_curves.png", "Lightweight CNN")
    paper_style = pd.DataFrame([
        {"Metric": "Accuracy", "Value": metrics["accuracy"]}, {"Metric": "Precision (weighted)", "Value": metrics["precision_weighted"]},
        {"Metric": "Recall (weighted)", "Value": metrics["recall_weighted"]}, {"Metric": "F1 score (weighted)", "Value": metrics["f1_weighted"]},
        {"Metric": "Sensitivity (macro)", "Value": metrics["sensitivity"]}, {"Metric": "Specificity (macro)", "Value": metrics["specificity"]},
        {"Metric": "ROC-AUC (macro, one-vs-rest)", "Value": metrics["roc_auc_macro"]}, {"Metric": "Matthews correlation coefficient", "Value": metrics["mcc"]},
        {"Metric": "False positive rate", "Value": metrics["false_positive_rate"]}, {"Metric": "False negative rate", "Value": metrics["false_negative_rate"]},
        {"Metric": "Log loss", "Value": metrics["log_loss"]}])
    paper_style.to_csv(out / "metrics_table.csv", index=False)
    reporting.save_table_image(paper_style, out / "metrics_table.png", title="Test-set metrics")
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "classes": class_names, "metrics": metrics}, indent=2))

    md = ("# Results (generated by `python -m src.train`)\n\n"
          f"Classes: {k} | Train/Val/Test images: {len(tr)}/{len(va)}/{len(te)}\n\n## Test-set metrics\n\n"
          + reporting.df_to_markdown(paper_style)
          + f"\nParameters: {n_params / 1e6:.2f} M | FP32 size: {n_params * 4 / 1e6:.1f} MB | "
            f"CPU latency (1 thread): {metrics['cpu_latency_ms']:.1f} ms per image\n\n## Per-class metrics\n\n"
          + reporting.df_to_markdown(per_class))
    (out / "RESULTS.md").write_text(md)
    print(f"\nTest accuracy {metrics['accuracy']:.4f} | results saved in {out}/")


if __name__ == "__main__":
    main()
