"""
DCGAN — Generator and Discriminator.

Architecture follows Radford et al. (2015) "Unsupervised Representation
Learning with Deep Convolutional Generative Adversarial Networks".

Trained on CelebA to generate 64 × 64 face images from a 100-dim noise vector.
"""

import torch
import torch.nn as nn


# ── Weight initialisation ──────────────────────────────────────────────────────

def weights_init(m: nn.Module) -> None:
    """Apply normal weight init to Conv, ConvTranspose, and BatchNorm layers."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps a latent noise vector z ∈ ℝ^latent_dim → RGB image ∈ [-1, 1]^(3×64×64).

    Architecture (nf = n_filters):
        z (100)  →  ConvT 4×4 stride 1  →  nf×8 (4×4)
                 →  ConvT 4×4 stride 2  →  nf×4 (8×8)
                 →  ConvT 4×4 stride 2  →  nf×2 (16×16)
                 →  ConvT 4×4 stride 2  →  nf   (32×32)
                 →  ConvT 4×4 stride 2  →  3    (64×64)
    Each layer (except last): ConvTranspose2d → BatchNorm → ReLU
    Final layer:               ConvTranspose2d → Tanh
    """

    def __init__(self, latent_dim: int = 100, n_filters: int = 64, num_channels: int = 3):
        super().__init__()
        nf = n_filters

        self.net = nn.Sequential(
            # --- Block 1: z → 4×4 feature maps ---
            nn.ConvTranspose2d(latent_dim, nf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),

            # --- Block 2: 4 → 8 ---
            nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),

            # --- Block 3: 8 → 16 ---
            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),

            # --- Block 4: 16 → 32 ---
            nn.ConvTranspose2d(nf * 2, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            # --- Block 5: 32 → 64 (output) ---
            nn.ConvTranspose2d(nf, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: noise tensor of shape (N, latent_dim, 1, 1)
        Returns:
            fake images of shape (N, 3, 64, 64) in [-1, 1]
        """
        return self.net(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Maps an RGB image ∈ [-1, 1]^(3×64×64) → scalar ∈ (0, 1) (real probability).

    Architecture (nf = n_filters):
        3 (64×64)   →  Conv 4×4 stride 2  →  nf    (32×32)
                    →  Conv 4×4 stride 2  →  nf×2  (16×16)
                    →  Conv 4×4 stride 2  →  nf×4  (8×8)
                    →  Conv 4×4 stride 2  →  nf×8  (4×4)
                    →  Conv 4×4 stride 1  →  1     (1×1)
    Each layer (except first and last): Conv2d → BatchNorm → LeakyReLU(0.2)
    First layer:  Conv2d → LeakyReLU(0.2)  (no BN on first layer per paper)
    Final layer:  Conv2d → Sigmoid
    """

    def __init__(self, n_filters: int = 64, num_channels: int = 3):
        super().__init__()
        nf = n_filters

        self.net = nn.Sequential(
            # --- Block 1: 64 → 32 (no BatchNorm on first layer) ---
            nn.Conv2d(num_channels, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # --- Block 2: 32 → 16 ---
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # --- Block 3: 16 → 8 ---
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # --- Block 4: 8 → 4 ---
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # --- Block 5: 4 → 1 (classification head) ---
            nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (N, 3, 64, 64)
        Returns:
            scalar probabilities of shape (N, 1, 1, 1)
        """
        return self.net(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
