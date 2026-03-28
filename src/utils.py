"""
Utility helpers for training and evaluation.
"""

import os
import random
import glob

import numpy as np
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Running average ───────────────────────────────────────────────────────────

class AverageMeter:
    """Track a running mean over values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    path: str,
    keep_last: int = 3,
    checkpoint_dir: str = "./checkpoints",
) -> None:
    """
    Save a checkpoint and remove old ones, keeping only `keep_last` files.

    Args:
        state:          Dict to serialise (epoch, model/optim state dicts …).
        path:           Full path for the new checkpoint file.
        keep_last:      Number of most-recent checkpoints to retain.
        checkpoint_dir: Directory to scan for old checkpoints.
    """
    torch.save(state, path)

    # Prune old checkpoints
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    )
    while len(checkpoints) > keep_last:
        os.remove(checkpoints.pop(0))


# ── Image helpers ─────────────────────────────────────────────────────────────

def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalised tensor in [-1, 1] back to [0, 1]."""
    return tensor * 0.5 + 0.5
