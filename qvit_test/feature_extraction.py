"""MNIST loading and patch tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST dataloader configuration.
@dataclass
class PatchConfig:
    image_size: int = 28
    patch_size: int = 7
    in_channels: int = 1
    embed_dim: int = 64

    @property
    def num_patches(self) -> int:
        # Number of non-overlapping patches per image.
        return (self.image_size // self.patch_size) ** 2

# CNN-based patch tokenizer block.
class PatchTokenizerCNN(nn.Module):
    """Split image into patches and use a small CNN to embed each patch."""

    def __init__(self, config: PatchConfig):
        super().__init__()
        self.config = config
        # Unfold to extract non-overlapping patches.
        self.unfold = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)
        # Simple CNN to embed each patch.
        self.cnn = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, config.embed_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        patches = self.unfold(x)  # (B, C*P*P, N)
        bsz, _, num_patches = patches.shape
        p = self.config.patch_size
        patches = patches.transpose(1, 2).contiguous()  # (B, N, C*P*P)
        patches = patches.view(bsz * num_patches, self.config.in_channels, p, p) # (B*N, C, P, P)
        emb = self.cnn(patches).view(bsz * num_patches, self.config.embed_dim) # (B*N, D)
        emb = emb.view(bsz, num_patches, self.config.embed_dim) # (B, N, D)
        return emb

# Simple patch embedding classifier for pretraining. Used to train the PatchTokenizerCNN only.
class PatchEmbeddingClassifier(nn.Module):
    """A simple classifier to pretrain the patch CNN using mean pooled tokens."""

    def __init__(self, tokenizer: PatchTokenizerCNN, num_classes: int = 10):
        super().__init__()
        self.tokenizer = tokenizer
        self.head = nn.Linear(tokenizer.config.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)

# MNIST dataloader utility.
def get_mnist_dataloaders(
    batch_size: int = 64,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    '''
    MNIST dataloader utility.
    
    :param batch_size: The batch size for the dataloaders.
    :type batch_size: int
    :param data_dir: The directory to store the dataset.
    :type data_dir: str
    :return: The train and test DataLoaders for MNIST dataset.
    :rtype: Tuple[DataLoader, DataLoader]
    '''
    # Standard MNIST normalization.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# train the patch tokenizer using a simple classifier head. Returns the trained tokenizer.
def pretrain_patch_tokenizer(
    tokenizer: PatchTokenizerCNN,
    train_loader: DataLoader,
    device: torch.device | str = "cpu",
    epochs: int = 1,
    lr: float = 1e-3,
) -> PatchTokenizerCNN:
    '''
    Train the patch tokenizer using a simple classifier head. Returns the trained tokenizer.
    
    :param tokenizer: The PatchTokenizerCNN instance to be trained.
    :type tokenizer: PatchTokenizerCNN
    :param train_loader: The DataLoader for training data.
    :type train_loader: DataLoader
    :param device: The device to run the training on.
    :type device: torch.device | str
    :param epochs: The number of epochs for training.
    :type epochs: int
    :param lr: The learning rate for the optimizer.
    :type lr: float
    :return: The trained PatchTokenizerCNN instance.
    :rtype: PatchTokenizerCNN
    '''
    # Create a simple model that combines the tokenizer with a linear head for classification.
    model = PatchEmbeddingClassifier(tokenizer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # Training loop for pretraining the patch tokenizer.
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Return the trained tokenizer.
    return tokenizer
