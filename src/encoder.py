"""ResNet-32 encoder (CIFAR variant, aligned with LDAM-DRW baseline).

Returns a 4x4x64 feature map flattened to 16 context tokens (64-dim each).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class ResNet32(nn.Module):
    dim = 64

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers = []
        ch, n = 16, 5  # 6n+2 with n=5 -> ResNet-32
        for stage, out_ch in enumerate((16, 32, 64)):
            stride = 1 if stage == 0 else 2
            for b in range(n):
                layers.append(BasicBlock(ch, out_ch, stride if b == 0 else 1))
                ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)                       # B,64,4,4
        feats = out.flatten(2).transpose(1, 2)       # B,16,64  context tokens
        pooled = out.mean((2, 3))                    # B,64     global feature
        return feats, pooled

    def classify(self, x):
        _, pooled = self.forward(x)
        return self.fc(pooled)
