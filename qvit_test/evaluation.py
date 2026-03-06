"""Evaluation helpers for ViT/QVIT models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .feature_extraction import PatchTokenizerCNN
from .qvit import QVIT
from .vit import ViT

# Evaluation configuration: device only; QVIT uses config.json for attention_filter / grover / topk.
@dataclass
class EvalConfig:
    device: str = "cpu"

# Evaluation function for both ViT and QVIT, returning accuracy and loss.
def evaluate_tokens_classifier(
    model: nn.Module,
    tokenizer: PatchTokenizerCNN,
    test_loader: DataLoader,
    config: EvalConfig,
) -> Dict[str, float]:
    '''
    Evaluate a token-based classifier (ViT or QVIT) on the given test dataset.
    
    :param model: The token-based classifier model (ViT or QVIT).
    :type model: nn.Module
    :param tokenizer: The tokenizer to convert images to tokens.
    :type tokenizer: PatchTokenizerCNN
    :param test_loader: The DataLoader for the test dataset.
    :type test_loader: DataLoader
    :param config: The evaluation configuration.
    :type config: EvalConfig
    :return: A dictionary containing the accuracy and loss on the test dataset.
    :rtype: Dict[str, float]
    '''
    # Evaluation mode for deterministic behavior.
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
            # Sanitize tokenizer output for stable evaluation.
            tokens = torch.nan_to_num(tokenizer(images), nan=0.0, posinf=1e4, neginf=-1e4)

            if isinstance(model, QVIT):
                logits, _ = model(tokens)
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

# Evaluation function specifically for ViT, utilizing standard token classification without Grover search.
def evaluate_vit(
    model: ViT,
    tokenizer: PatchTokenizerCNN,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    '''
    Evaluate a ViT model on the given test dataset without using Grover search.
    
    :param model: The ViT model to evaluate.
    :type model: ViT
    :param tokenizer: The tokenizer to convert images to tokens.
    :type tokenizer: PatchTokenizerCNN
    :param test_loader: The DataLoader for the test dataset.
    :type test_loader: DataLoader
    :param device: The device to run the evaluation on.
    :type device: str
    :return: A dictionary containing the accuracy and loss on the test dataset.
    :rtype: Dict[str, float]
    '''
    return evaluate_tokens_classifier(model, tokenizer, test_loader, EvalConfig(device=device))

# QVIT evaluation: attention_filter and grover/topk params are read from config.json.
def evaluate_qvit(
    model: QVIT,
    tokenizer: PatchTokenizerCNN,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    return evaluate_tokens_classifier(model, tokenizer, test_loader, EvalConfig(device=device))
