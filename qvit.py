"""QVIT core modules: Grover-filtered attention and transformer blocks."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .qiskit_grover import grover_mask
from .vit import ViTConfig


def grover_search_filter(
    attn_probs: torch.Tensor,
<<<<<<< HEAD
    threshold: float = 0.0482,
    use_qiskit: bool = True,
    max_qubits: int = 4,
    shots: int | None = None,
    enable_filter: bool = True,
) -> torch.Tensor:
    """Grover-search-inspired filtering of attention probabilities."""
    
=======
    threshold: float = 0.0482, # Threshold for attention score filtering.
    use_qiskit: bool = True, # Whether to use Qiskit for Grover simulation. If False, uses classical thresholding.
    max_qubits: int = 4, # Maximum qubits for Grover simulation.
    shots: int | None = None, # Number of shots for Grover simulation. If None, uses 2x sequence length.
    enable_filter: bool = True, # Whether to enable Grover search filtering. If False, no filtering is applied. and a mask of all True is returned.
) -> torch.Tensor:
    '''
    Grover-search-inspired filtering of attention probabilities. Returns a boolean mask of selected indices.
    
    :param attn_probs: Attention probabilities tensor of shape (B, H, S, S).
    :type attn_probs: torch.Tensor
    :param threshold: Threshold for attention score filtering.
    :type threshold: float
    :param use_qiskit: Whether to use Qiskit for Grover simulation. If False, uses classical thresholding.
    :type use_qiskit: bool
    :param max_qubits: Maximum qubits for Grover simulation.
    :type max_qubits: int
    :param shots: Number of shots for Grover simulation. If None, uses 2x sequence length.
    :type shots: int | None
    :param enable_filter: Whether to enable Grover search filtering. If False, no filtering is applied. and a mask of all True is returned.
    :type enable_filter: bool
    :return: Boolean mask tensor of shape (B, H, S, S) indicating selected indices.
    :rtype: torch.Tensor
    '''

>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    # Fast path: no filtering.
    if not enable_filter:
        return torch.ones_like(attn_probs, dtype=torch.bool)

    # Classical thresholding fallback when Qiskit is disabled.
    if not use_qiskit:
<<<<<<< HEAD
        mask = attn_probs > threshold
        # 保险：确保每行至少一个 True
        any_selected = mask.any(dim=-1, keepdim=True)
        if (~any_selected).any():
            fallback_idx = attn_probs.argmax(dim=-1, keepdim=True)
            fallback = torch.zeros_like(mask)
            fallback.scatter_(-1, fallback_idx, True)
            mask = torch.where(any_selected, mask, fallback)
        return mask

    # Qiskit path
    bsz, heads, seq_len, _ = attn_probs.shape
    
    # 序列太长，回退到经典方法
    if seq_len > (2**max_qubits):
        mask = attn_probs > threshold
        any_selected = mask.any(dim=-1, keepdim=True)
        if (~any_selected).any():
            fallback_idx = attn_probs.argmax(dim=-1, keepdim=True)
            fallback = torch.zeros_like(mask)
            fallback.scatter_(-1, fallback_idx, True)
            mask = torch.where(any_selected, mask, fallback)
        return mask

    effective_shots = seq_len * 2 if shots is None else min(shots, seq_len * 2)
    mask = torch.zeros_like(attn_probs, dtype=torch.bool)
    
    failed_count = 0
    
=======
        return attn_probs > threshold

    # attn_probs: (B, H, S, S)
    bsz, heads, seq_len, _ = attn_probs.shape
    if seq_len > (2**max_qubits):
        return attn_probs > threshold

    seq_len = attn_probs.shape[-1]
    # Use 2x sequence length as a lightweight default for Grover shots.
    effective_shots = seq_len * 2 if shots is None else min(shots, seq_len * 2)

    # Prepare mask tensor.
    mask = torch.zeros_like(attn_probs, dtype=torch.bool)
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    for b in range(bsz):
        for h in range(heads):
            for i in range(seq_len):
                row = attn_probs[b, h, i].detach().cpu().tolist()
<<<<<<< HEAD
                
                # 保险：如果 row 全相同，直接返回全 True
                if len(set(row)) == 1:
                    mask[b, h, i] = torch.ones(seq_len, dtype=torch.bool, device=attn_probs.device)
                    continue
                
                try:
                    keep = grover_mask(row, threshold, max_qubits=max_qubits, shots=effective_shots)
                except Exception as e:
                    # Qiskit 失败，回退到经典方法
                    keep = [v > threshold for v in row]
                    failed_count += 1
                
                # 关键保险：如果返回全 False，强制选最大值
                if not any(keep):
                    max_idx = row.index(max(row))
                    keep = [False] * len(row)
                    keep[max_idx] = True
                    failed_count += 1
                
                mask[b, h, i] = torch.tensor(keep, dtype=torch.bool, device=attn_probs.device)
    
    if failed_count > 0:
        print(f"[Grover Filter] Fallback triggered for {failed_count}/{bsz*heads*seq_len} positions")

    # 最终保险：再次确保每行至少一个 True
    any_selected = mask.any(dim=-1, keepdim=True)
    if (~any_selected).any():
        print(f"[Grover Filter] Final fallback: some rows all False, using argmax")
=======
                try:
                    keep = grover_mask(
                        row,
                        threshold,
                        max_qubits=max_qubits,
                        shots=effective_shots,
                    )
                except Exception:
                    raise RuntimeError("Grover simulation failed.")
                mask[b, h, i] = torch.tensor(keep, dtype=torch.bool, device=attn_probs.device)
            
    # Ensure at least one token is selected per row to avoid all -inf softmax
    any_selected = mask.any(dim=-1, keepdim=True)
    if (~any_selected).any():
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        fallback_idx = attn_probs.argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(mask)
        fallback.scatter_(-1, fallback_idx, True)
        mask = torch.where(any_selected, mask, fallback)
<<<<<<< HEAD
    
=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    return mask

# Single self-attention layer with Grover-filtered attention.
class GroverFilteredAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3) # Q, K, V projection
<<<<<<< HEAD
        self.proj = nn.Linear(embed_dim, embed_dim)    # Output projection
=======
        self.proj = nn.Linear(embed_dim, embed_dim) # Output projection
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
<<<<<<< HEAD
        threshold: float | None = 0.0482,   # 支持 None：动态阈值
=======
        threshold: float = 0.0482,
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        use_qiskit: bool = True,
        max_qubits: int = 4,
        shots: int | None = None,
        enable_filter: bool = True,
<<<<<<< HEAD
        percentile: float = 0.5,            # TODO1: 动态阈值用的百分位
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            out: [B, S, D]
            filtered_attn: [B, H, S, S]
            used_threshold: float，本次前向实际使用的阈值
        """
=======
    ) -> Tuple[torch.Tensor, torch.Tensor]:
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # First pass attention (traditional ViT).
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
<<<<<<< HEAD
        # 防止 softmax 溢出
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)
        attn_probs = attn_logits.softmax(dim=-1)

        # === 动态阈值：threshold 为 None -> 用本 batch percentile ===
        if threshold is None:
            with torch.no_grad():
                probs_clean = torch.nan_to_num(
                    attn_probs.detach().cpu(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                flat = probs_clean.reshape(-1)
                
                # 计算分位数，但限制范围防止极端值
                raw_threshold = torch.quantile(flat, percentile).item()
                used_threshold = max(0.01, min(raw_threshold, 0.5))  # 限制在 [0.01, 0.5]
                
                # 打印警告如果阈值被裁剪
                if raw_threshold != used_threshold:
                    print(f"[Threshold Clipped] raw={raw_threshold:.4f} -> used={used_threshold:.4f}")
        else:
            used_threshold = float(threshold)

        # Grover-search-inspired filtering on attention probabilities.
        selected = grover_search_filter(
            attn_probs,
            threshold=used_threshold,
=======
        attn_probs = attn_logits.softmax(dim=-1)

        # Grover-search-inspired filtering on attention probabilities.
        selected = grover_search_filter(
            attn_probs,
            threshold=threshold,
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
            use_qiskit=use_qiskit,
            max_qubits=max_qubits,
            shots=shots,
            enable_filter=enable_filter,
        )

        # Second pass: only compute attention on selected indices.
        filtered_logits = attn_logits.masked_fill(~selected, float("-inf"))
        filtered_attn = filtered_logits.softmax(dim=-1)
        filtered_attn = self.dropout(filtered_attn)

        out = filtered_attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        out = self.proj(out)
<<<<<<< HEAD
        return out, filtered_attn, used_threshold
=======
        return out, filtered_attn
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81

# Single transformer block with Grover-filtered attention.
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
<<<<<<< HEAD
        threshold: float | None = 0.0482,
=======
        threshold: float = 0.0482,
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        use_qiskit: bool = True,
        max_qubits: int = 4,
        shots: int | None = None,
        enable_filter: bool = True,
<<<<<<< HEAD
        percentile: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        attn_out, attn, used_threshold = self.attn(
=======
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
            self.norm1(x),
            threshold=threshold,
            use_qiskit=use_qiskit,
            max_qubits=max_qubits,
            shots=shots,
            enable_filter=enable_filter,
<<<<<<< HEAD
            percentile=percentile,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn, used_threshold
=======
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81

# ViT model with Grover-filtered attention.
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
<<<<<<< HEAD
        # === TODO1: 阈值统计相关 ===
        self.record_threshold_stats: bool = False
        self.threshold_sum: float = 0.0
        self.threshold_count: int = 0
        self.current_batch_threshold: float | None = None
        self.learned_threshold: float | None = None
=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81

    def forward(
        self,
        tokens: torch.Tensor,
<<<<<<< HEAD
        threshold: float | None = 0.0482,   # None → 动态；float → 固定
=======
        threshold: float = 0.0482,
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        use_qiskit: bool = True,
        max_qubits: int = 4,
        shots: int | None = None,
        enable_filter: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        last_attn = None
<<<<<<< HEAD
        last_threshold = None
        for block in self.blocks:
            x, last_attn, last_threshold = block(
=======
        for block in self.blocks:
            x, last_attn = block(
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
                x,
                threshold=threshold,
                use_qiskit=use_qiskit,
                max_qubits=max_qubits,
                shots=shots,
                enable_filter=enable_filter,
<<<<<<< HEAD
                percentile=0.9,
            )

        # 记录本 batch 最后一层使用的阈值（用于 TODO1 统计）
        self.current_batch_threshold = last_threshold

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, last_attn
        
=======
            )

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, last_attn
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
