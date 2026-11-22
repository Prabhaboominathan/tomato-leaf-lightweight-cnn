"""Training utilities: lazy-loading dataset with augmentation, training loop, prediction."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LeafDataset(Dataset):
    """Reads each image from disk, resizes to size x size (bilinear) and scales pixels to [0, 1].
    Training images get rotation (25 deg), shift (10 %), shear, zoom (20 %) and horizontal flips."""

    def __init__(self, paths, labels, train: bool, size: int = 224):
        self.paths, self.labels, self.size = list(paths), list(labels), size
        ops = []
        if train:
            ops += [transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=11.3)]
        ops.append(transforms.ToTensor())                       # [0, 255] -> [0, 1], as in the pre-processing step
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        return self.tf(img), int(self.labels[i])


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, epochs, lr, weight_decay, device, name, class_weights=None, patience=0):
    """Adam (L2 weight decay) + cross-entropy + LR reduction on plateau. Best validation-loss epoch is restored.
    patience = 0 trains for all epochs; patience > 0 enables early stopping."""
    weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, device, val_criterion)
        scheduler.step(vl)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(float(v))
        print(f"[{name}] epoch {ep:03d}  loss {tl:.4f} acc {ta:.4f}  |  val_loss {vl:.4f} val_acc {va:.4f}")
        if vl < best_loss - 1e-5:
            best_loss, wait = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if patience and wait >= patience:
                print(f"[{name}] early stopping at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    return np.concatenate([F.softmax(model(x.to(device)), dim=1).cpu().numpy() for x, _ in loader])
