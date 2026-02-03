from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .qiskit_grover import grover_mask
from .vit import ViTConfig


def grover_search_filter(
    attn_probs: torch.Tensor,
    threshold: float = 0.1,
    use_qiskit: bool = True,
    max_qubits: int = 4,
    shots: int | None = None,
) -> torch.Tensor:
    """Select attention indices using Grover search (Qiskit simulation) when possible."""

    if not use_qiskit:
        return attn_probs > threshold

    # attn_probs: (B, H, S, S)
    bsz, heads, seq_len, _ = attn_probs.shape
    if seq_len > (2**max_qubits):
        return attn_probs > threshold

    seq_len = attn_probs.shape[-1]
    effective_shots = seq_len * 2 if shots is None else min(shots, seq_len * 2)

    mask = torch.zeros_like(attn_probs, dtype=torch.bool)
    for b in range(bsz):
        for h in range(heads):
            for i in range(seq_len):
                row = attn_probs[b, h, i].detach().cpu().tolist()
                try:
                    keep = grover_mask(
                        row,
                        threshold,
                        max_qubits=max_qubits,
                        shots=effective_shots,
                    )
                except Exception:
                    keep = [v > threshold for v in row]
                mask[b, h, i] = torch.tensor(keep, dtype=torch.bool, device=attn_probs.device)
    # Ensure at least one token is selected per row to avoid all -inf softmax
    any_selected = mask.any(dim=-1, keepdim=True)
    if (~any_selected).any():
        fallback_idx = attn_probs.argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(mask)
        fallback.scatter_(-1, fallback_idx, True)
        mask = torch.where(any_selected, mask, fallback)
    return mask


class GroverFilteredAttention(nn.Module):
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
        threshold: float = 0.1,
        use_qiskit: bool = True,
        max_qubits: int = 4,
        shots: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # First pass attention (traditional ViT)
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_logits.softmax(dim=-1)

        # Grover-search-inspired filtering on attention probabilities
        selected = grover_search_filter(
            attn_probs,
            threshold=threshold,
            use_qiskit=use_qiskit,
            max_qubits=max_qubits,
            shots=shots,
        )

        # Second pass: only compute attention on selected indices
        filtered_logits = attn_logits.masked_fill(~selected, float("-inf"))
        filtered_attn = filtered_logits.softmax(dim=-1)
        filtered_attn = self.dropout(filtered_attn)

        out = filtered_attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        out = self.proj(out)
        return out, filtered_attn


class QVITBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = GroverFilteredAttention(
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
        threshold: float = 0.1,
        use_qiskit: bool = True,
        max_qubits: int = 4,
        shots: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(
            self.norm1(x),
            threshold=threshold,
            use_qiskit=use_qiskit,
            max_qubits=max_qubits,
            shots=shots,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn


class QVIT(nn.Module):
    """ViT with Grover-search-filtered attention."""

    def __init__(self, num_patches: int, config: Optional[ViTConfig] = None):
        super().__init__()
        self.config = config or ViTConfig()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.config.embed_dim)
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [QVITBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.head = nn.Linear(self.config.embed_dim, self.config.num_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        threshold: float = 0.1,
        use_qiskit: bool = True,
        max_qubits: int = 4,
        shots: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        last_attn = None
        for block in self.blocks:
            x, last_attn = block(
                x,
                threshold=threshold,
                use_qiskit=use_qiskit,
                max_qubits=max_qubits,
                shots=shots,
            )

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, last_attn
