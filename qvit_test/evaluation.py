from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .feature_extraction import PatchTokenizerCNN
from .qvit import QVIT
from .vit import ViT


@dataclass
class EvalConfig:
    device: str = "cpu"
    use_qiskit: bool = True
    threshold: float = 0.1
    max_qubits: int = 4
    shots: int | None = None


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate_tokens_classifier(
    model: nn.Module,
    tokenizer: PatchTokenizerCNN,
    test_loader: DataLoader,
    config: EvalConfig,
) -> Dict[str, float]:
    model.eval()
    tokenizer.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            tokens = torch.nan_to_num(tokenizer(images), nan=0.0, posinf=1e4, neginf=-1e4)

            if isinstance(model, QVIT):
                logits, _ = model(
                    tokens,
                    threshold=config.threshold,
                    use_qiskit=config.use_qiskit,
                    max_qubits=config.max_qubits,
                    shots=config.shots,
                )
            elif isinstance(model, ViT):
                logits, _ = model(tokens)
            else:
                logits = model(tokens)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return {
        "accuracy": correct / max(1, total),
        "loss": total_loss / max(1, total),
    }


def evaluate_vit(
    model: ViT,
    tokenizer: PatchTokenizerCNN,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    config = EvalConfig(device=device, use_qiskit=False)
    return evaluate_tokens_classifier(model, tokenizer, test_loader, config)


def evaluate_qvit(
    model: QVIT,
    tokenizer: PatchTokenizerCNN,
    test_loader: DataLoader,
    device: str = "cpu",
    threshold: float = 0.1,
    use_qiskit: bool = True,
    max_qubits: int = 4,
    shots: int | None = None,
) -> Dict[str, float]:
    config = EvalConfig(
        device=device,
        use_qiskit=use_qiskit,
        threshold=threshold,
        max_qubits=max_qubits,
        shots=shots,
    )
    return evaluate_tokens_classifier(model, tokenizer, test_loader, config)
