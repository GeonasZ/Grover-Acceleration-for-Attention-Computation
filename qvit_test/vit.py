"""Minimal ViT implementation for patch token inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class ViTConfig:
    embed_dim: int = 24
    num_heads: int = 2
    mlp_ratio: int = 2
    num_layers: int = 2
    num_classes: int = 10
    dropout: float = 0.1

# Standard multi-head self-attention block.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Standard scaled dot-product attention.
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))
        attn = attn_logits.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        out = self.proj(out)
        return out, attn

# Single transformer block with standard multi-head self-attention.
class TransformerBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(
            config.embed_dim, config.num_heads, config.dropout
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * config.mlp_ratio, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn

# ViT model that consumes precomputed patch tokens.
class ViT(nn.Module):
    """A minimal ViT that consumes precomputed patch tokens."""

    def __init__(self, num_patches: int, config: ViTConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        # Prepend CLS token and add positional embeddings.
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        last_attn = None
        for block in self.blocks:
            x, last_attn = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, last_attn
