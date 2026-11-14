"""Smoke test for DCGAN Generator and Discriminator."""
import torch
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Generator, Discriminator


def test_dcgan_generator_and_discriminator_shapes():
    latent_dim = 100
    batch_size = 2
    net_g = Generator(latent_dim=latent_dim, n_filters=16)
    net_d = Discriminator(n_filters=16)

    # 1. Generator forward
    z = torch.randn(batch_size, latent_dim, 1, 1)
    fake = net_g(z)
    assert fake.shape == (batch_size, 3, 64, 64)
    assert fake.min() >= -1.0 and fake.max() <= 1.0

    # 2. Discriminator forward
    pred = net_d(fake)
    assert pred.shape == (batch_size, 1, 1, 1)
    assert pred.min() >= 0.0 and pred.max() <= 1.0
