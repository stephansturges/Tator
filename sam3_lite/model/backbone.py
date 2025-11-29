from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNetBackbone(nn.Module):
    """Lightweight feature extractor returning a single stride-32 map."""

    def __init__(self, out_dim: int = 256, pretrained: bool = False):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        layers = list(backbone.children())[:-2]
        self.stem = nn.Sequential(*layers)
        in_channels = 2048
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,3,H,W]
        feats = self.stem(images)  # [B,2048,H/32,W/32]
        feats = self.proj(feats)   # [B,C,H/32,W/32]
        return feats
