"""
Inference script — generate face images from a trained DCGAN checkpoint.

Usage:
    python src/generate.py --checkpoint checkpoints/checkpoint_epoch_0050.pth
    python src/generate.py --checkpoint checkpoints/checkpoint_epoch_0050.pth \\
        --num-images 64 --output results/generated.png --temperature 0.9
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.model import Generator
from configs.config import DCGANConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate faces with trained DCGAN")
    p.add_argument("--checkpoint",   type=str, required=True,   help="Path to .pth checkpoint")
    p.add_argument("--num-images",   type=int, default=64,      help="Number of images to generate")
    p.add_argument("--output",       type=str, default="results/generated.png")
    p.add_argument("--nrow",         type=int, default=8,       help="Images per row in the grid")
    p.add_argument("--temperature",  type=float, default=1.0,   help="Scale noise std dev (diversity control)")
    p.add_argument("--seed",         type=int, default=None,    help="Random seed for reproducibility")
    return p.parse_args()


def generate(args: argparse.Namespace) -> None:
    cfg = DCGANConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────
    G = Generator(cfg.latent_dim, cfg.n_filters, cfg.num_channels).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("G", checkpoint)
    # Strip DataParallel prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    G.load_state_dict(state_dict)
    G.eval()

    print(f"Loaded Generator from: {args.checkpoint}")

    # ── Sample ────────────────────────────────────────────────────────────
    with torch.no_grad():
        noise = torch.randn(args.num_images, cfg.latent_dim, 1, 1, device=device)
        noise = noise * args.temperature
        fake_imgs = G(noise).cpu()

    # Denormalise: [-1, 1] → [0, 1]
    fake_imgs = fake_imgs * 0.5 + 0.5

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(fake_imgs, str(output_path), nrow=args.nrow, padding=2)

    print(f"Saved {args.num_images} generated images → {output_path}")


if __name__ == "__main__":
    generate(parse_args())
