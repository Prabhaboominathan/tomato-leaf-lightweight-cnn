"""Lightweight CNN built from scratch: stem + 5 bottleneck blocks (1x1 expand, 3x3 depthwise,
Squeeze-and-Excitation, 1x1 project) + global average pooling + dense head. Swish activation (SiLU)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_act(kind: str) -> nn.Module:
    return nn.SiLU() if kind == "swish" else nn.ReLU(inplace=True)


class SqueezeExcite(nn.Module):
    """Global average pooling -> two fully connected layers (reduction 1/16) -> sigmoid gate."""

    def __init__(self, channels: int, ratio: int = 16, act: str = "swish"):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = make_act(act)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(s))))


class Bottleneck(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, expand: int, use_se: bool, act: str):
        super().__init__()
        mid = c_in * expand
        self.expand = nn.Sequential(nn.Conv2d(c_in, mid, 1, bias=False), nn.BatchNorm2d(mid), make_act(act))
        self.depthwise = nn.Sequential(nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                                       nn.BatchNorm2d(mid), make_act(act))
        self.se = SqueezeExcite(mid, 16, act) if use_se else nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out))   # no activation
        self.skip = stride == 1 and c_in == c_out

    def forward(self, x):
        y = self.project(self.se(self.depthwise(self.expand(x))))
        return x + y if self.skip else y


# (output channels, stride) of the five bottleneck blocks: 112 -> 112 -> 56 -> 28 -> 14 -> 7 for a 224 x 224 input
STAGES = [(64, 1), (128, 2), (256, 2), (384, 2), (512, 2)]


class LightweightCNN(nn.Module):
    def __init__(self, num_classes: int = 10, expand: int = 6, use_se: bool = True, activation: str = "swish",
                 dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32),
                                  make_act(activation))
        blocks, c = [], 32
        for c_out, stride in STAGES:
            blocks.append(Bottleneck(c, c_out, stride, expand, use_se, activation))
            c = c_out
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(c, 128), make_act(activation), nn.Dropout(dropout),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))        # logits; softmax at prediction time


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
