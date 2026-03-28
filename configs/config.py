"""
DCGANConfig — central hyperparameter dataclass.

All values match the Kaggle training run (GPU T4 x2, CelebA dataset).
"""
from dataclasses import dataclass


@dataclass
class DCGANConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    dataset: str = "celeba"                  # torchvision dataset name
    data_dir: str = "./data"                 # root dir for dataset download
    image_size: int = 64                     # spatial resolution after crop
    num_workers: int = 2

    # ── Model ─────────────────────────────────────────────────────────────
    latent_dim: int = 100                    # dimension of noise vector z
    n_filters: int = 64                      # base feature-map width (nf)
    num_channels: int = 3                    # RGB

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 2e-4
    beta1: float = 0.5                       # Adam β₁ (paper recommendation)
    beta2: float = 0.999                     # Adam β₂

    # ── Logging / checkpoints ─────────────────────────────────────────────
    log_interval: int = 50                   # log every N batches
    sample_interval: int = 5                 # save sample grid every N epochs
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    use_comet: bool = True                   # set False to disable Comet ML

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42
