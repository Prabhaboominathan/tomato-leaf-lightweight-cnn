"""Metrics tables and figures (no torch needed): confusion matrix, ROC, accuracy/loss curves, tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize


def summary_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy plus macro / weighted precision, recall, F1 and macro one-vs-rest ROC-AUC."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    if y_prob is not None:
        try:
            out["roc_auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                                 labels=list(range(y_prob.shape[1])))
        except ValueError:
            out["roc_auc_macro"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def per_class_table(y_true, y_pred, class_names) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support, followed by macro and weighted averages."""
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"Class": class_names, "Precision": p, "Recall": r, "F1-score": f, "Support": s})
    for avg, name in (("macro", "Macro avg"), ("weighted", "Weighted avg")):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        df.loc[len(df)] = [name, ap, ar, af, int(s.sum())]
    df.loc[len(df)] = ["Accuracy", np.nan, np.nan, accuracy_score(y_true, y_pred), int(s.sum())]
    return df


def plot_confusion_matrix(y_true, y_pred, class_names, path, normalize=False, title=None):
    k = len(class_names)
    big = k > 12
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    shown = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm
    size = (0.34 * k + 3.0, 0.34 * k + 2.6) if big else (5.5, 4.8)
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fs = 6 if big else 10
    ax.set_xticks(range(k), class_names, rotation=90 if big else 30, ha="center" if big else "right", fontsize=fs)
    ax.set_yticks(range(k), class_names, fontsize=fs)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title or ("Normalised confusion matrix" if normalize else "Confusion matrix"))
    if not big:
        thresh = shown.max() / 2.0
        for i in range(k):
            for j in range(k):
                txt = f"{shown[i, j]:.2f}" if normalize else f"{int(shown[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", color="white" if shown[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(y_true, y_prob, class_names, path, title="ROC curves (one-vs-rest)"):
    k = len(class_names)
    big = k > 12
    y_bin = label_binarize(y_true, classes=list(range(k)))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    all_fpr, tprs = np.linspace(0, 1, 200), []
    for c in range(k):
        if y_bin[:, c].sum() == 0:                      # class absent from this test set
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        if big:
            ax.plot(fpr, tpr, lw=0.6, alpha=0.35, color="tab:blue")
        else:
            ax.plot(fpr, tpr, lw=1.6, label=f"{class_names[c]} (AUC = {auc(fpr, tpr):.3f})")
        tprs.append(np.interp(all_fpr, fpr, tpr))
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro average (AUC = {auc(all_fpr, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title + (f" ({k} classes, thin lines)" if big else ""))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def history_table(history: dict) -> pd.DataFrame:
    """Epoch table: loss, accuracy, validation loss, validation accuracy."""
    n = len(history["train_loss"])
    return pd.DataFrame({"Epoch": range(1, n + 1), "Loss": history["train_loss"], "Accuracy": history["train_acc"],
                         "Validation loss": history["val_loss"], "Validation accuracy": history["val_acc"]})


def plot_history(history: dict, path, title):
    """Training / validation accuracy and loss curves (paper Fig. 6 and Fig. 7)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_acc"], label="Training")
    axes[0].plot(epochs, history["val_acc"], label="Validation")
    axes[0].set_title(f"{title}: accuracy")
    axes[1].plot(epochs, history["train_loss"], label="Training")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title(f"{title}: loss")
    for ax, ylab in zip(axes, ("Accuracy", "Loss")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_table_image(df: pd.DataFrame, path, title=None, float_fmt="{:.4f}"):
    cells = [[float_fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
              else ("" if isinstance(v, (float, np.floating)) else str(v)) for v in row]
             for row in df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(1.9 * len(df.columns), 0.5 * (len(df) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=list(df.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dbe8f7")
            cell.set_text_props(weight="bold")
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame, float_fmt="{:.4f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    rows = ""
    for row in df.itertuples(index=False):
        cells = [float_fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                 else ("" if isinstance(v, (float, np.floating)) else str(v)) for v in row]
        rows += "| " + " | ".join(cells) + " |\n"
    return head + rows


def save_all_for_model(name, y_true, y_pred, y_prob, class_names, out_dir):
    out = Path(out_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, out / f"{name}_confusion_matrix.png",
                          title=f"{name}: confusion matrix")
    plot_confusion_matrix(y_true, y_pred, class_names, out / f"{name}_confusion_matrix_normalized.png",
                          normalize=True, title=f"{name}: normalised confusion matrix")
    plot_roc(y_true, y_prob, class_names, out / f"{name}_roc_curve.png", title=f"{name}: ROC curves")
    table = per_class_table(y_true, y_pred, class_names)
    table.to_csv(out / f"{name}_per_class_report.csv", index=False)
    save_table_image(table, out / f"{name}_per_class_report.png", title=f"{name}: per-class test metrics")
    return table


def extended_metrics(y_true, y_pred, y_prob) -> dict:
    """Sensitivity, specificity, MCC, FPR, FNR and log loss (macro-averaged over classes)."""
    k = y_prob.shape[1]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    sens = float(np.mean(tp / np.clip(tp + fn, 1, None)))
    spec = float(np.mean(tn / np.clip(tn + fp, 1, None)))
    return {"sensitivity": sens, "specificity": spec, "false_positive_rate": 1 - spec,
            "false_negative_rate": 1 - sens, "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_prob, labels=list(range(k))))}
