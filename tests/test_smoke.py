"""Checks that need no dataset and no GPU:  python -m tests.test_smoke"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src import reporting
from src.compare_runs import build_table
from src.data import clean_name, list_images, split_indices


def make_folder(d, rng):
    for cls in ["Tomato___Bacterial_spot", "Tomato___healthy", "Tomato___Late_blight", "Potato___healthy"]:
        (Path(d) / cls).mkdir()
        for i in range(30):
            Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(Path(d) / cls / f"{i}.jpg")


def test_data():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        make_folder(d, rng)
        paths, labels, classes = list_images(d, "Tomato", max_per_class=20)
        assert classes == ["Bacterial spot", "Late blight", "Healthy"] and len(paths) == 60       # Potato ignored, 20 per class
        tr, va, te = split_indices(labels, 0.15, 0.15, 0)
        assert len(te) == 9 and len(va) == 9 and len(tr) == 42 and not set(tr) & set(te)
    assert clean_name("Tomato___Tomato_Yellow_Leaf_Curl_Virus") == "Tomato Yellow Leaf Curl Virus"


def test_metrics_and_reports():
    rng = np.random.default_rng(1)
    k, n = 10, 500
    y = rng.integers(0, k, n)
    prob = rng.dirichlet(np.ones(k), n)
    prob[np.arange(n), y] += 2.0
    prob /= prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    ext = reporting.extended_metrics(y, pred, prob)
    perfect = reporting.extended_metrics(y, y, np.eye(k)[y] * 0.99 + 0.001)
    assert abs(perfect["mcc"] - 1.0) < 1e-9 and perfect["false_negative_rate"] == 0
    assert 0 < ext["specificity"] <= 1 and ext["log_loss"] > 0
    with tempfile.TemporaryDirectory() as d:
        reporting.save_all_for_model("model", y, pred, prob, [f"C{i}" for i in range(k)], d)
        assert (Path(d) / "model_confusion_matrix.png").exists() and (Path(d) / "model_roc_curve.png").exists()


def test_ablation_table():
    with tempfile.TemporaryDirectory() as d:
        dirs = []
        for name, acc, args in [("a", .95, {}), ("b", .94, {"no_se": True}), ("c", .945, {"activation": "relu"})]:
            p = Path(d) / name
            p.mkdir()
            (p / "metrics.json").write_text(json.dumps({"args": args, "metrics": {"accuracy": acc, "f1_weighted": acc, "parameters_millions": 4.6}}))
            dirs.append(p)
        df = build_table(dirs)
        assert list(df["Variant"]) == ["Full model (SE + Swish)", "Without SE", "Swish replaced by ReLU"]
        assert abs(df["Change in accuracy"].iloc[1] + 0.01) < 1e-9


if __name__ == "__main__":
    test_data()
    test_metrics_and_reports()
    test_ablation_table()
    print("All smoke tests passed")
