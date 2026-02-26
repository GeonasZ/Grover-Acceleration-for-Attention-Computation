"""
Simple patch tokenizer: flatten each patch to pixel*channel and project with one linear layer to embed_dim.
Keeps the same interface as PatchTokenizerCNN in feature_extraction for drop-in replacement.
"""

from __future__ import annotations

import torch
from torch import nn

from .feature_extraction import PatchConfig


class SimplePatchTokenizer(nn.Module):
    """
    Split image into patches, flatten each patch to (C*P*P,) and project with Linear(C*P*P, embed_dim).
    No CNN, no pretrained tokenizer.
    """

    def __init__(self, config: PatchConfig):
        super().__init__()
        self.config = config
        self.unfold = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)
        patch_dim = config.in_channels * config.patch_size * config.patch_size
        self.proj = nn.Linear(patch_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        patches = self.unfold(x)  # (B, C*P*P, N)
        patches = patches.transpose(1, 2).contiguous()  # (B, N, C*P*P)
        emb = self.proj(patches)  # (B, N, embed_dim)
        return emb
