"""QVIT core modules: Grover-filtered attention and transformer blocks."""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
from torch import nn

from .qiskit_grover import grover_mask
from .vit import ViTConfig


def _build_local_neighborhood_mask(
    num_patches: int | None,
    seq_len: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Build (S, S) boolean mask: for each query position i, set mask[i, j] = True
    iff j is "local" to i: (1) CLS query (row 0) sees all positions; (2) every
    query sees CLS key (column 0); (3) for patch query i, j in self + 2D neighbor
    patches within radius. Used so that Grover runs only on non-local positions.
    """
    if radius < 0 or num_patches is None or seq_len != num_patches + 1:
        return None
    grid = int(math.sqrt(num_patches))
    if grid * grid != num_patches:
        return None

    S = seq_len
    local = torch.eye(S, dtype=torch.bool, device=device)
    # CLS query (row 0): attend to all keys.
    local[0, :] = True
    # Every query attends to CLS key (column 0).
    local[:, 0] = True
    # For each patch query i (1..S-1), mark self and 2D neighbors as local.
    for i in range(1, S):
        idx = i - 1
        r, c = idx // grid, idx % grid
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    j = rr * grid + cc + 1
                    local[i, j] = True
    return local


def grover_search_filter(
    attn_probs: torch.Tensor,
    threshold: float = 0.0482,  # Base threshold used as fallback / for inference.
    use_qiskit: bool = True,  # Whether to use Qiskit for Grover simulation. If False, uses classical thresholding.
    max_qubits: int = 5,  # Need 5 qubits for 17 tokens (16 patches + 1 CLS); 4 qubits only supports 16.
    shots: int | None = None,  # Number of shots for Grover simulation. If None, uses 2x sequence length.
    enable_filter: bool = True,  # If False, no filtering is applied and a mask of all True is returned.
    percentile: float = 0.9,  # Row-wise percentile used in training to derive dynamic thresholds.
    use_row_percentile: bool = False,  # If True, use per-row percentile thresholds instead of a fixed threshold.
    history_threshold: torch.Tensor | None = None,  # Optional running threshold (0-D tensor) updated during training.
    update_history: bool = False,  # If True, update history_threshold with EMA of row thresholds.
    use_history_threshold: bool = False,  # If True, ignore row percentiles and use history_threshold (for inference).
    local_mask: torch.Tensor | None = None,  # Optional (S, S): positions already True (e.g. self+neighbors). Grover runs only on the rest.
) -> torch.Tensor:
    """
    Grover-search-inspired filtering of attention probabilities. Returns a boolean mask of selected indices.

    If local_mask (S, S) is provided: for each query row, positions where local_mask is True are
    kept as-is; Grover (or classical threshold) is applied only to the remaining "remote" positions,
    reducing Grover search space while guaranteeing local neighborhood is always attended.
    """

    # Fast path: no filtering.
    if not enable_filter:
        return torch.ones_like(attn_probs, dtype=torch.bool)

    # attn_probs: (B, H, S, S)
    bsz, heads, seq_len, _ = attn_probs.shape
    device = attn_probs.device

    # Default shots for full-row Grover when no local_mask.
    default_shots = seq_len * 2 if shots is None else min(shots, seq_len * 2)

    # --- Compute thresholds (per row, over full sequence) ---
    if use_history_threshold and history_threshold is not None and history_threshold.item() > 0.0:
        th = float(history_threshold.item())
        thresholds = torch.full((bsz, heads, seq_len), th, device=device, dtype=attn_probs.dtype)
    elif use_row_percentile:
        thresholds = torch.quantile(attn_probs, percentile, dim=-1)
        if history_threshold is not None and update_history:
            batch_mean_qt = thresholds.detach().float().mean().item()
            momentum = 0.1
            history_threshold.mul_(1.0 - momentum).add_(momentum * batch_mean_qt)
    else:
        thresholds = torch.full((bsz, heads, seq_len), threshold, device=device, dtype=attn_probs.dtype)

    # Initialize mask: if local_mask given, start with it (broadcast to B, H, S, S); else zeros.
    if local_mask is not None:
        # local_mask: (S, S) -> (1, 1, S, S)
        local_bhw = local_mask.unsqueeze(0).unsqueeze(0).expand(bsz, heads, -1, -1)
        mask = local_bhw.clone()
    else:
        mask = torch.zeros_like(attn_probs, dtype=torch.bool)
        if use_qiskit and seq_len > (2**max_qubits):
            raise ValueError("Sequence length too large for Grover simulation.")

    # Indices where we need to run Grover/threshold: for each row i, j where local_mask[i,j] is False (or all j if no local_mask).
    if local_mask is not None:
        # remote_mask[b,h,i,j] = True means (i,j) is not local, need to decide by Grover/threshold
        remote_mask = ~local_bhw
        n_remote_per_row = remote_mask.sum(dim=-1)  # (B, H, S)
    else:
        remote_mask = torch.ones_like(attn_probs, dtype=torch.bool)
        n_remote_per_row = torch.full((bsz, heads, seq_len), seq_len, device=device, dtype=torch.long)

    if not use_qiskit:
        # Classical: set remote positions by threshold; local positions already True in mask.
        remote_selected = attn_probs > thresholds.unsqueeze(-1)
        mask = mask | (remote_mask & remote_selected)
    else:
        for b in range(bsz):
            for h in range(heads):
                for i in range(seq_len):
                    n_remote = n_remote_per_row[b, h, i].item()
                    if n_remote == 0:
                        continue
                    remote_idx = remote_mask[b, h, i].nonzero(as_tuple=True)[0]
                    scores_full = attn_probs[b, h, i].detach().cpu().tolist()
                    scores_remote = [scores_full[j] for j in remote_idx.tolist()]
                    t = thresholds[b, h, i].item()
                    n_qubits_needed = math.ceil(math.log2(n_remote)) if n_remote > 0 else 0
                    if n_qubits_needed > max_qubits:
                        # Fallback: classical threshold on remote indices only
                        keep_remote = [s > t for s in scores_remote]
                    else:
                        effective_shots = default_shots if shots is None else min(shots, n_remote * 2)
                        keep_remote = grover_mask(scores_remote, t, max_qubits=max_qubits, shots=effective_shots)
                    for idx, j in enumerate(remote_idx.tolist()):
                        if keep_remote[idx]:
                            mask[b, h, i, j] = True

    # Ensure at least one token is selected per row to avoid all -inf softmax.
    any_selected = mask.any(dim=-1, keepdim=True)
    if (~any_selected).any():
        fallback_idx = attn_probs.argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(mask)
        fallback.scatter_(-1, fallback_idx, True)
        mask = torch.where(any_selected, mask, fallback)
    return mask

# Single self-attention layer with Grover-filtered attention.
class GroverFilteredAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_patches: int | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3) # Q, K, V projection
        self.proj = nn.Linear(embed_dim, embed_dim) # Output projection
        self.dropout = nn.Dropout(dropout)
        self.num_patches = num_patches
        # History of effective thresholds (running EMA of per-row percentiles).
        self.register_buffer("history_threshold", torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        threshold: float = 0.0482,
        use_qiskit: bool = True,
        max_qubits: int = 5,
        shots: int | None = None,
        enable_filter: bool = True,
        percentile: float = 0.9,
        neighbor_radius: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # First pass attention (traditional ViT).
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_logits.softmax(dim=-1)

        # Local mask: for each query, self + neighbor patches (and CLS) are always True; Grover runs only on the rest.
        local_mask = _build_local_neighborhood_mask(
            self.num_patches, seq_len, neighbor_radius, x.device
        )

        # Grover (or classical threshold) only on non-local positions; local positions stay True.
        selected = grover_search_filter(
            attn_probs,
            threshold=threshold,
            use_qiskit=use_qiskit,
            max_qubits=max_qubits,
            shots=shots,
            enable_filter=enable_filter,
            percentile=percentile,
            use_row_percentile=self.training,
            history_threshold=self.history_threshold,
            update_history=self.training,
            use_history_threshold=not self.training,
            local_mask=local_mask,
        )

        # Second pass: only compute attention on selected indices.
        filtered_logits = attn_logits.masked_fill(~selected, float("-inf"))
        filtered_attn = filtered_logits.softmax(dim=-1)
        filtered_attn = self.dropout(filtered_attn)

        out = filtered_attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        out = self.proj(out)
        return out, filtered_attn

# Single transformer block with Grover-filtered attention.
class QVITBlock(nn.Module):
    def __init__(self, config: ViTConfig, num_patches: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = GroverFilteredAttention(
            config.embed_dim,
            config.num_heads,
            config.dropout,
            num_patches=num_patches,
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
        threshold: float = 0.0482,
        use_qiskit: bool = True,
        max_qubits: int = 5,
        shots: int | None = None,
        enable_filter: bool = True,
        percentile: float = 0.9,
        neighbor_radius: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(
            self.norm1(x),
            threshold=threshold,
            use_qiskit=use_qiskit,
            max_qubits=max_qubits,
            shots=shots,
            enable_filter=enable_filter,
            percentile=percentile,
            neighbor_radius=neighbor_radius,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn

# ViT model with Grover-filtered attention.
class QVIT(nn.Module):
    """ViT with Grover-search-filtered attention."""

    def __init__(self, num_patches: int, config: Optional[ViTConfig] = None):
        super().__init__()
        self.config = config or ViTConfig()
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.config.embed_dim)
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [QVITBlock(self.config, num_patches=num_patches) for _ in range(self.config.num_layers)]
        )
        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.head = nn.Linear(self.config.embed_dim, self.config.num_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        threshold: float = 0.0482,
        use_qiskit: bool = True,
        max_qubits: int = 5,
        shots: int | None = None,
        enable_filter: bool = True,
        percentile: float = 0.9,
        neighbor_radius: int = 1,
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
                enable_filter=enable_filter,
                percentile=percentile,
                neighbor_radius=neighbor_radius,
            )

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, last_attn
