"""Training/evaluation entry point for ViT/QVIT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from qvit_test.evaluation import evaluate_qvit, evaluate_vit
from qvit_test.feature_extraction import PatchConfig, PatchTokenizerCNN, get_mnist_dataloaders
from qvit_test.qvit import QVIT
from qvit_test.vit import ViT, ViTConfig
from qvit_test.simple_tokenizer import SimplePatchTokenizer

@dataclass
class TrainConfig:
    """Run configuration for training and evaluation."""
    batch_size: int = 64
<<<<<<< HEAD
    epochs: int = 3
=======
    epochs: int = 10
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    lr: float = 1e-3
    data_dir: str = "./data"
    device: str = "cpu"
    train_qvit: bool = True
    eval_qvit: bool = True
    qvit_use_grover: bool = True
    qvit_enable_filter: bool = True
    qvit_filter_start_epoch: int = 0

# Train for one epoch. Returns average loss.
<<<<<<< HEAD
# Train for one epoch. Returns average loss.
def _train_epoch(
    model: nn.Module,
    tokenizer: PatchTokenizerCNN,
=======
def _train_epoch(
    model: nn.Module,
    tokenizer: PatchTokenizerCNN|SimplePatchTokenizer,
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_qiskit: bool,
    freeze_tokenizer: bool = False,
    enable_filter: bool = True,
) -> float:
    # Switch model and tokenizer to the desired training mode.
    model.train()
    if freeze_tokenizer:
        tokenizer.eval()
        for param in tokenizer.parameters():
            param.requires_grad_(False)
    else:
        tokenizer.train()
        for param in tokenizer.parameters():
            param.requires_grad_(True)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Sanitize tokenizer output to avoid NaN/Inf propagation.
        tokens = torch.nan_to_num(tokenizer(images), nan=0.0, posinf=1e4, neginf=-1e4)
<<<<<<< HEAD

        if isinstance(model, QVIT):
            # 训练阶段：threshold=None → 在 QVIT 内部按当前 batch percentile 动态计算阈值
            logits, attn = model(
                tokens,
                threshold=None,
                use_qiskit=use_qiskit,
                enable_filter=enable_filter,
            )

            # TODO1：旁路统计当前 batch 的阈值
            if getattr(model, "record_threshold_stats", False):
                with torch.no_grad():
                    batch_th = getattr(model, "current_batch_threshold", None)
                    if batch_th is not None:
                        model.threshold_sum += batch_th
                        model.threshold_count += 1
        else:
            logits, attn = model(tokens)
=======
        if isinstance(model, QVIT):
            logits, _ = model(
                tokens,
                use_qiskit=use_qiskit,
                enable_filter=enable_filter,
            )
        else:
            logits, _ = model(tokens)
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81

        # Sanitize logits before loss computation.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
<<<<<<< HEAD
        # 新增：梯度裁剪，防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    return total_loss / max(1, total)

#
def train_and_evaluate(
    train_cfg: TrainConfig,
    vit_cfg: ViTConfig | None = None,
) -> Dict[str, Dict[str, float]]:
    # Use default ViT config if not provided.
    vit_cfg = vit_cfg or ViTConfig()
    patch_cfg = PatchConfig(embed_dim=vit_cfg.embed_dim)

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=train_cfg.batch_size,
        data_dir=train_cfg.data_dir,
    )

    # Shared patch tokenizer (frozen during QVIT training).
<<<<<<< HEAD
    tokenizer = PatchTokenizerCNN(patch_cfg).to(train_cfg.device)
=======
    # tokenizer = PatchTokenizerCNN(patch_cfg).to(train_cfg.device)
    tokenizer = SimplePatchTokenizer(patch_cfg).to(train_cfg.device)
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    vit = ViT(num_patches=patch_cfg.num_patches, config=vit_cfg).to(train_cfg.device)

    vit_opt = torch.optim.AdamW(
        list(vit.parameters()) + list(tokenizer.parameters()), lr=train_cfg.lr
    )
<<<<<<< HEAD

=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
    qvit = None
    qvit_opt = None
    if train_cfg.train_qvit:
        qvit = QVIT(num_patches=patch_cfg.num_patches, config=vit_cfg).to(train_cfg.device)
        qvit_opt = torch.optim.AdamW(
            list(qvit.parameters()) + list(tokenizer.parameters()), lr=train_cfg.lr
        )
<<<<<<< HEAD
        # TODO1: 初始化阈值统计
        qvit.record_threshold_stats = True
        qvit.threshold_sum = 0.0
        qvit.threshold_count = 0
        qvit.current_batch_threshold = None
        qvit.learned_threshold = None
=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81

    for epoch in range(1, train_cfg.epochs + 1):
        print(f"Epoch {epoch}/{train_cfg.epochs} - ViT training...")
        _train_epoch(
            vit,
            tokenizer,
            train_loader,
            vit_opt,
            train_cfg.device,
            use_qiskit=False,
            freeze_tokenizer=False,
            enable_filter=True,
        )
<<<<<<< HEAD

=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        # Optionally train QVIT with gradual filtering.
        if train_cfg.train_qvit and qvit is not None and qvit_opt is not None:
            print(f"Epoch {epoch}/{train_cfg.epochs} - QVIT training...")
            enable_filter = (
                train_cfg.qvit_enable_filter
                and epoch >= train_cfg.qvit_filter_start_epoch
            )
            _train_epoch(
                qvit,
                tokenizer,
                train_loader,
                qvit_opt,
                train_cfg.device,
                use_qiskit=train_cfg.qvit_use_grover,
                freeze_tokenizer=True,
                enable_filter=enable_filter,
            )

<<<<<<< HEAD
        # 训练结束：处理 QVIT 的 threshold 和调试
    if qvit is not None:
        # 1. 计算 learned_threshold
        if qvit.threshold_count > 0:
            qvit.learned_threshold = qvit.threshold_sum / qvit.threshold_count
            print(f"[Threshold Learning] learned_threshold = {qvit.learned_threshold:.6f}")
        else:
            qvit.learned_threshold = None
            print("[Threshold Learning] No thresholds collected; will use default 0.0482 in eval.")

        # ============================================================
        # 【插入：详细调试代码】
        # ============================================================
        print("\n" + "="*50)
        print("DEBUG: Detailed QVIT attention check")
        print("="*50)
        
        qvit.eval()
        tokenizer.eval()
        
        with torch.no_grad():
            # 取一个 batch
            images, labels = next(iter(test_loader))
            images = images.to(train_cfg.device)
            tokens = torch.nan_to_num(tokenizer(images), nan=0.0, posinf=1e4, neginf=-1e4)
            
            print(f"\n1. Tokens stats:")
            print(f"   Shape: {tokens.shape}")
            print(f"   Range: [{tokens.min():.4f}, {tokens.max():.4f}]")
            print(f"   Mean: {tokens.mean():.4f}, Std: {tokens.std():.4f}")
            print(f"   NaN: {torch.isnan(tokens).any()}, Inf: {torch.isinf(tokens).any()}")
            
            # 手动构造输入
            bsz = tokens.shape[0]
            cls = qvit.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, tokens], dim=1)
            x = x + qvit.pos_embed
            
            print(f"\n2. After pos_embed:")
            print(f"   Range: [{x.min():.4f}, {x.max():.4f}]")
            
            # 测试第一个 block，经典模式（无 Qiskit，无过滤）
            block = qvit.blocks[0]
            x_norm = block.norm1(x)
            
            print(f"\n3. CLASSICAL mode (use_qiskit=False, enable_filter=False):")
            attn_out, attn, used_th = block.attn(
                x_norm, 
                threshold=None, 
                use_qiskit=False, 
                enable_filter=False
            )
            print(f"   Threshold: {used_th:.6f}")
            print(f"   Attention range: [{attn.min():.4f}, {attn.max():.4f}]")
            print(f"   Attention mean: {attn.mean():.4f}")
            print(f"   NaN ratio: {torch.isnan(attn).float().mean():.4f}")
            
            # 测试 Qiskit 模式
            print(f"\n4. QUANTUM mode (use_qiskit=True, enable_filter=True):")
            x_norm2 = block.norm1(x)  # 重新 norm
            attn_out2, attn2, used_th2 = block.attn(
                x_norm2, 
                threshold=None, 
                use_qiskit=True, 
                enable_filter=True
            )
            print(f"   Threshold: {used_th2:.6f}")
            print(f"   Attention range: [{attn2.min():.4f}, {attn2.max():.4f}]")
            print(f"   Attention mean: {attn2.mean():.4f}")
            print(f"   NaN ratio: {torch.isnan(attn2).float().mean():.4f}")
            
            # 5. 原有的 quick check（简化版）
            print(f"\n5. Full model forward (use_qiskit=False, enable_filter=False):")
            _, attn_full = qvit(
                tokens,
                use_qiskit=False,
                enable_filter=False,
                threshold=None,
            )
            if attn_full is not None:
                nan_ratio = torch.isnan(attn_full).float().mean().item()
                attn_clean = torch.nan_to_num(attn_full, nan=0.0, posinf=1.0, neginf=0.0)
                print(f"   NaN ratio: {nan_ratio:.4f}, Mean: {attn_clean.mean():.4f}")
            
        print("\n" + "="*50)
        print("END DEBUG")
        print("="*50 + "\n")
        # ============================================================

    
    vit_metrics = evaluate_vit(vit, tokenizer, test_loader, device=train_cfg.device)
    results: Dict[str, Dict[str, float]] = {"vit": vit_metrics}

    if train_cfg.eval_qvit and qvit is not None:
        # 决定评估时使用的阈值
        if qvit.learned_threshold is not None:
            th_eval = qvit.learned_threshold
            print(f"[Eval] Using learned_threshold = {th_eval:.6f} for QVIT.")
        else:
            th_eval = 0.0482
            print(f"[Eval] Using default threshold = {th_eval:.4f} for QVIT.")

=======
    # Quick attention distribution check on a single batch.
    if qvit is not None:
        qvit.eval()
        tokenizer.eval()
        with torch.no_grad():
            images, _ = next(iter(test_loader))
            images = images.to(train_cfg.device)
            tokens = torch.nan_to_num(tokenizer(images), nan=0.0, posinf=1e4, neginf=-1e4)
            _, attn = qvit(
                tokens,
                use_qiskit=False,
                enable_filter=False,
            )
            if attn is not None:
                nan_ratio = torch.isnan(attn).float().mean().item()
                attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
                flat = attn.flatten()
                median = torch.quantile(flat, 0.5).item()
                topk = min(8, attn.shape[-1])
                topk_vals, _ = attn.topk(topk, dim=-1)
                mean_topk = topk_vals.mean().item()
                mean_all = attn.mean().item()
                print(
                    f"QVIT attention mean: {mean_all:.4f}, top-{topk} mean: {mean_topk:.4f}, "
                    f"median: {median:.4f}, nan ratio: {nan_ratio:.4f}"
                )

    vit_metrics = evaluate_vit(vit, tokenizer, test_loader, device=train_cfg.device)
    results: Dict[str, Dict[str, float]] = {"vit": vit_metrics}
    if train_cfg.eval_qvit and qvit is not None:
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
        qvit_metrics = evaluate_qvit(
            qvit,
            tokenizer,
            test_loader,
            device=train_cfg.device,
<<<<<<< HEAD
            threshold=th_eval,
=======
>>>>>>> cb50362427ccf2d21217aac6bfaa6c28f5229c81
            use_qiskit=train_cfg.qvit_use_grover,
        )
        results["qvit"] = qvit_metrics

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device {device}.")
    results = train_and_evaluate(TrainConfig(device=device))
    print(results)
