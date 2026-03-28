"""
DCGAN Training Script — CelebA Face Generation.

Usage:
    python src/train.py                         # default config
    python src/train.py --epochs 100 --lr 3e-4

Experiment tracking: Comet ML (set COMET_API_KEY env var or disable with --no-comet).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from src.model import Generator, Discriminator, weights_init
from src.dataset import build_dataloaders
from src.utils import set_seed, AverageMeter, save_checkpoint
from configs.config import DCGANConfig


# ── Labels ────────────────────────────────────────────────────────────────────
REAL_LABEL = 1.0
FAKE_LABEL = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCGAN on CelebA")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--latent-dim", type=int,   default=None)
    p.add_argument("--data-dir",   type=str,   default=None)
    p.add_argument("--no-comet",   action="store_true", help="Disable Comet ML logging")
    p.add_argument("--resume",     type=str,   default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def train(cfg: DCGANConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    # ── Comet ML ──────────────────────────────────────────────────────────
    experiment = None
    if cfg.use_comet and os.getenv("COMET_API_KEY"):
        from comet_ml import Experiment
        experiment = Experiment(project_name="facegenerationdcgan", auto_metric_logging=False)
        experiment.log_parameters(vars(cfg))

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, _ = build_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # ── Models ────────────────────────────────────────────────────────────
    G = Generator(cfg.latent_dim, cfg.n_filters, cfg.num_channels).to(device)
    D = Discriminator(cfg.n_filters, cfg.num_channels).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    print(f"Generator params:     {G.module.num_parameters() if hasattr(G,'module') else G.num_parameters():,}")
    print(f"Discriminator params: {D.module.num_parameters() if hasattr(D,'module') else D.num_parameters():,}")

    # ── Loss & Optimisers ─────────────────────────────────────────────────
    criterion = nn.BCELoss()

    opt_G = optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    # Fixed noise for visualisation (consistent across epochs)
    fixed_noise = torch.randn(64, cfg.latent_dim, 1, 1, device=device)

    # ── Output dirs ───────────────────────────────────────────────────────
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train()

        d_loss_meter = AverageMeter()
        g_loss_meter = AverageMeter()
        dx_meter     = AverageMeter()   # D(x)   — should stay near 0.5
        dgz1_meter   = AverageMeter()   # D(G(z)) before G update
        dgz2_meter   = AverageMeter()   # D(G(z)) after  G update

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs}", leave=False)

        for batch_idx, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            # ── Train Discriminator ───────────────────────────────────────
            D.zero_grad()

            # Real batch
            label_real = torch.full((bsz,), REAL_LABEL, device=device)
            out_real = D(real_imgs).view(-1)
            loss_D_real = criterion(out_real, label_real)
            loss_D_real.backward()
            D_x = out_real.mean().item()

            # Fake batch
            noise = torch.randn(bsz, cfg.latent_dim, 1, 1, device=device)
            fake_imgs = G(noise)
            label_fake = torch.full((bsz,), FAKE_LABEL, device=device)
            out_fake = D(fake_imgs.detach()).view(-1)
            loss_D_fake = criterion(out_fake, label_fake)
            loss_D_fake.backward()
            D_G_z1 = out_fake.mean().item()

            loss_D = loss_D_real + loss_D_fake
            opt_D.step()

            # ── Train Generator ───────────────────────────────────────────
            G.zero_grad()
            label_gen = torch.full((bsz,), REAL_LABEL, device=device)
            out_gen = D(fake_imgs).view(-1)
            loss_G = criterion(out_gen, label_gen)
            loss_G.backward()
            D_G_z2 = out_gen.mean().item()
            opt_G.step()

            # ── Metrics ───────────────────────────────────────────────────
            d_loss_meter.update(loss_D.item(), bsz)
            g_loss_meter.update(loss_G.item(), bsz)
            dx_meter.update(D_x, bsz)
            dgz1_meter.update(D_G_z1, bsz)
            dgz2_meter.update(D_G_z2, bsz)

            if batch_idx % cfg.log_interval == 0:
                pbar.set_postfix(
                    D=f"{d_loss_meter.avg:.3f}",
                    G=f"{g_loss_meter.avg:.3f}",
                    Dx=f"{dx_meter.avg:.3f}",
                    DGz=f"{dgz2_meter.avg:.3f}",
                )

                if experiment:
                    step = (epoch - 1) * len(train_loader) + batch_idx
                    experiment.log_metric("loss_D",  d_loss_meter.avg, step=step)
                    experiment.log_metric("loss_G",  g_loss_meter.avg, step=step)
                    experiment.log_metric("D_x",     dx_meter.avg,     step=step)
                    experiment.log_metric("D_G_z1",  dgz1_meter.avg,   step=step)
                    experiment.log_metric("D_G_z2",  dgz2_meter.avg,   step=step)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"D Loss: {d_loss_meter.avg:.4f} | G Loss: {g_loss_meter.avg:.4f} | "
            f"D(x): {dx_meter.avg:.4f} | D(G(z)): {dgz2_meter.avg:.4f}"
        )

        # ── Sample grid ───────────────────────────────────────────────────
        if epoch % cfg.sample_interval == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
            save_image(
                samples * 0.5 + 0.5,   # denormalise to [0, 1]
                f"{cfg.results_dir}/epoch_{epoch:04d}.png",
                nrow=8,
                padding=2,
            )
            G.train()

        # ── Checkpoint ────────────────────────────────────────────────────
        save_checkpoint(
            {"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
             "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()},
            path=f"{cfg.checkpoint_dir}/checkpoint_epoch_{epoch:04d}.pth",
            keep_last=3,
            checkpoint_dir=cfg.checkpoint_dir,
        )

    print("Training complete.")
    if experiment:
        experiment.end()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg = DCGANConfig()

    if args.epochs:     cfg.epochs        = args.epochs
    if args.batch_size: cfg.batch_size    = args.batch_size
    if args.lr:         cfg.learning_rate = args.lr
    if args.latent_dim: cfg.latent_dim    = args.latent_dim
    if args.data_dir:   cfg.data_dir      = args.data_dir
    if args.no_comet:   cfg.use_comet     = False

    train(cfg)
