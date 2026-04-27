from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = (
    "SoftSPDConv",
    "LHFE",
    "GAC3k2"
)

class SoftSPDLayer(nn.Module):
    def __init__(self, c1):
        super().__init__()
        init = torch.tensor([
            [[[[[1/2, 1/6, 1/6, 1/6]] * 4]]]
        ])
        self.weight_logits = nn.Parameter(init.repeat(1, c1, 1, 1, 1, 1))
        self.offsets = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 1), (0, 0), (1, 1), (1, 0)],
            [(1, 0), (0, 0), (0, 1), (1, 1)],
            [(1, 1), (0, 1), (1, 0), (0, 0)],
        ]

    def forward(self, x):
        weights = F.softmax(self.weight_logits, dim=-1)
        submaps = []
        for sub_i in range(4):
            vals = []
            for neigh_i, (dy, dx) in enumerate(self.offsets[sub_i]):
                vals.append(x[:, :, dy::2, dx::2] * weights[..., sub_i, neigh_i])
            submaps.append(sum(vals))
        return torch.cat(submaps, dim=1)


class SoftSPDConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.spd = SoftSPDLayer(c1)
        self.dwconv = DWConv(4 * c1, 4 * c1, k=3, s=1)
        self.conv = Conv(4 * c1, c2, k=1, s=1)

    def forward(self, x):
        return self.conv(self.dwconv(self.spd(x)))


class LHFE(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        self.dw1 = DWConv(c1, c1, k=7, s=1)
        self.dw2 = nn.Sequential(
            nn.Conv2d(c1, c1, 5, padding=2, groups=c1),
            nn.Conv2d(c1, c1, 1),
        )
        self.fuse = Conv(c1, c2, k=1, s=1)

    def forward(self, x):
        x1 = self.dw1(x)
        x2 = self.dw2(x)
        out = self.fuse(x1 + x2)
        return x + out


class GradientConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]], dtype=torch.float32
        )
        sobel_y = sobel_x.t()
        self.register_buffer(
            "weight_x",
            sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        )
        self.register_buffer(
            "weight_y",
            sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        )
        self.groups = channels

    def forward(self, x):
        gx = F.conv2d(x, self.weight_x, padding=1, groups=self.groups)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=self.groups)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class Bottleneck_GA(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.grad = GradientConv(c2)
        self.fuse = Conv(c2 * 2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        g = self.grad(y)
        y = self.fuse(torch.cat([y, g], dim=1))
        return x + y if self.add else y


class GAC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((n + 2) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck_GA(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

