"""
Grid-based single-shot object detector (YOLOv1-style, single box per cell).

Input:  (N, 3, img_size, img_size)
Output: (N, grid_size, grid_size, 5 + num_classes)
        channel 0        -> objectness, sigmoid
        channels 1:3      -> x, y offset within cell, sigmoid
        channels 3:5      -> w, h relative to full image, sigmoid
        channels 5:5+C     -> per-class confidence, sigmoid
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        return self.pool(x)


class SentinelDetector(nn.Module):
    """Lightweight Darknet-style backbone, downsamples by 32x (e.g. 416 -> 13)."""

    def __init__(self, num_classes: int, grid_size: int = 13):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_outputs = 5 + num_classes

        self.backbone = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512, pool=False),
        )
        self.head = nn.Conv2d(512, self.num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        raw = self.head(features)  # (N, num_outputs, S, S)
        raw = raw.permute(0, 2, 3, 1)  # (N, S, S, num_outputs)
        return torch.sigmoid(raw)
