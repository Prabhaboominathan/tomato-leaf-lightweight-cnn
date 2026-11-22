"""Build the ablation table from several training runs (no torch needed).

    python -m src.compare_runs results results_no_se results_relu --out ablation
The first directory is the reference (full model); the others show the change relative to it.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import reporting


def variant_name(args: dict) -> str:
    if args.get("no_se"):
        return "Without SE"
    if args.get("activation") == "relu":
        return "Swish replaced by ReLU"
    return "Full model (SE + Swish)"


def build_table(run_dirs) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        info = json.loads((Path(d) / "metrics.json").read_text())
        m = info["metrics"]
        rows.append({"Variant": variant_name(info["args"]), "Accuracy": m["accuracy"], "F1 (weighted)": m["f1_weighted"],
                     "Parameters (M)": m["parameters_millions"]})
    df = pd.DataFrame(rows)
    df["Change in accuracy"] = df["Accuracy"] - df["Accuracy"].iloc[0]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()
    df = build_table(args.runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out) + ".csv", index=False)
    reporting.save_table_image(df, str(out) + ".png", title="Ablation study")
    Path(str(out) + ".md").write_text(reporting.df_to_markdown(df))
    print(reporting.df_to_markdown(df))


if __name__ == "__main__":
    main()
