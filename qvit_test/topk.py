"""
Quantum-inspired top-k attention filter.
Selects top-k attention positions per query (with optional local mask); can be extended to quantum top-k later.
"""

from __future__ import annotations

import torch


def topk_search_filter(
    attn_probs: torch.Tensor,
    k: int = 8,
    enable_filter: bool = True,
    local_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Top-k attention filter: for each query row (each of the B×H×S rows), keep local
    positions (if local_mask given) plus top-k keys by attention score. Returns boolean
    mask (B, H, S, S). Selection is per row, not per head or per batch.
    """
    if not enable_filter:
        return torch.ones_like(attn_probs, dtype=torch.bool)

    bsz, heads, seq_len, _ = attn_probs.shape
    device = attn_probs.device

    if local_mask is not None:
        local_bhw = local_mask.unsqueeze(0).unsqueeze(0).expand(bsz, heads, -1, -1)
        mask = local_bhw.clone()
        remote_mask = ~local_bhw
    else:
        mask = torch.zeros_like(attn_probs, dtype=torch.bool)
        remote_mask = torch.ones_like(attn_probs, dtype=torch.bool)

    # Per query row: top-k key indices (vectorized over B, H, S).
    scores_remote = attn_probs.masked_fill(~remote_mask, float("-inf"))
    effective_k = min(k, attn_probs.size(-1))
    if effective_k <= 0:
        return mask
    _, topk_idx = scores_remote.topk(effective_k, dim=-1)
    scatter_one = torch.ones(bsz, heads, seq_len, effective_k, dtype=torch.bool, device=device)
    mask_remote = torch.zeros_like(attn_probs, dtype=torch.bool)
    mask_remote.scatter_(-1, topk_idx, scatter_one)
    mask = mask | mask_remote

    # Ensure at least one selected per row (if no local_mask we might have all False)
    if local_mask is None:
        any_selected = mask.any(dim=-1, keepdim=True)
        if (~any_selected).any():
            fallback_idx = attn_probs.argmax(dim=-1, keepdim=True)
            fallback = torch.zeros_like(mask)
            fallback.scatter_(-1, fallback_idx, True)
            mask = torch.where(any_selected, mask, fallback)
    return mask
