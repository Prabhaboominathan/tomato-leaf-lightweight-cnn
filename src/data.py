"""Finding images, class balancing and splitting (no torch needed)."""
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def clean_name(folder: str) -> str:
    """'Tomato___Bacterial_spot' -> 'Bacterial spot'."""
    name = folder.split("___")[-1].replace("_", " ").strip()
    return name[:1].upper() + name[1:]


def list_images(data_dir: str, prefix: str = "Tomato", max_per_class: int = 0, seed: int = 42) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Collect image paths from class folders whose name starts with `prefix` (empty = all folders)."""
    by_class = {}
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.parent.name.lower().startswith(prefix.lower()):
            by_class.setdefault(p.parent.name, []).append(p)
    if not by_class:
        raise RuntimeError(f"No images found under {data_dir} for folders starting with '{prefix}'. See data/README.md.")
    rng = random.Random(seed)
    folders = sorted(by_class)
    paths, labels = [], []
    for i, f in enumerate(folders):
        files = by_class[f]
        if max_per_class and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        paths += files
        labels += [i] * len(files)
    return paths, np.array(labels, dtype=np.int64), [clean_name(f) for f in folders]


def split_indices(labels, test_size=0.15, val_size=0.15, seed=42):
    """Stratified 70 / 15 / 15 split by default."""
    idx = np.arange(len(labels))
    trainval, test = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    train, val = train_test_split(trainval, test_size=val_size / (1.0 - test_size),
                                  stratify=labels[trainval], random_state=seed)
    return train, val, test
