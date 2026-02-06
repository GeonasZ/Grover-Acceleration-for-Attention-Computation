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


@dataclass
class TrainConfig:
    """Run configuration for training and evaluation."""
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    data_dir: str = "./data"
    device: str = "cpu"
    train_qvit: bool = True
    eval_qvit: bool = True
    qvit_use_grover: bool = True
    qvit_enable_filter: bool = True
    qvit_filter_start_epoch: int = 6

# Train for one epoch. Returns average loss.
def _train_epoch(
    model: nn.Module,
    tokenizer: PatchTokenizerCNN,
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
        if isinstance(model, QVIT):
            logits, _ = model(
                tokens,
                use_qiskit=use_qiskit,
                enable_filter=enable_filter,
            )
        else:
            logits, _ = model(tokens)

        # Sanitize logits before loss computation.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
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
    tokenizer = PatchTokenizerCNN(patch_cfg).to(train_cfg.device)
    vit = ViT(num_patches=patch_cfg.num_patches, config=vit_cfg).to(train_cfg.device)

    vit_opt = torch.optim.AdamW(
        list(vit.parameters()) + list(tokenizer.parameters()), lr=train_cfg.lr
    )
    qvit = None
    qvit_opt = None
    if train_cfg.train_qvit:
        qvit = QVIT(num_patches=patch_cfg.num_patches, config=vit_cfg).to(train_cfg.device)
        qvit_opt = torch.optim.AdamW(
            list(qvit.parameters()) + list(tokenizer.parameters()), lr=train_cfg.lr
        )

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
        qvit_metrics = evaluate_qvit(
            qvit,
            tokenizer,
            test_loader,
            device=train_cfg.device,
            use_qiskit=train_cfg.qvit_use_grover,
        )
        results["qvit"] = qvit_metrics

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device {device}.")
    results = train_and_evaluate(TrainConfig(device=device))
    print(results)
