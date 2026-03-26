"""
Dataset loading and preprocessing for CelebA face generation.

Transforms applied (matching Kaggle training run):
  - Resize to image_size × image_size
  - CenterCrop to image_size × image_size
  - ToTensor
  - Normalize to [-1, 1] (mean=0.5, std=0.5 per channel)
  - RandomHorizontalFlip (p=0.5)
  - RandomRotation (±15°)
  - ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_transforms(image_size: int = 64) -> transforms.Compose:
    """Return the full augmentation pipeline used during training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),          # → [-1, 1]
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ])


def build_dataloaders(
    data_dir: str = "./data",
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and return train / val DataLoaders for CelebA.

    Args:
        data_dir:    Root directory to store the dataset.
        image_size:  Target spatial resolution (default 64×64).
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader)
    """
    transform = get_transforms(image_size)
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CelebA(
        root=str(root),
        split="train",
        download=True,
        transform=transform,
    )
    val_dataset = datasets.CelebA(
        root=str(root),
        split="valid",
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset):,} images  |  Val: {len(val_dataset):,} images")
    return train_loader, val_loader
