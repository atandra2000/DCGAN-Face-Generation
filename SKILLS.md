# SKILLS.md — DCGAN-Face-Generation

> Skills for unconditional 64×64 face generation. Pair with the GAN
> debugging subagent in `Vision/DCGAN-Face-Generation/AGENTS.md`.

---

## Skill 1: Run a baseline DCGAN training

```bash
cd Vision/DCGAN-Face-Generation
python src/train.py --epochs 50 --batch_size 128 --lr 2e-4
```

CelebA must be pre-downloaded to `data/celeba/img_align_celeba/` (~1.4 GB,
202,599 images).

## Skill 2: Generate samples from a trained G

```bash
python src/generate.py --checkpoint checkpoints/netG_epoch_49.pt \
  --n_samples 64 --output samples/epoch_49.png
```

## Skill 3: Fix mode collapse

If G produces the same image regardless of z:

1. **Lower learning rate** to 1e-4 or 5e-5.
2. **Add noise** to D inputs:
   ```python
   real += 0.05 * torch.randn_like(real)
   ```
3. **Use label smoothing** on real labels:
   ```python
   real_labels = 0.9 * torch.ones_like(real_labels)
   ```
4. **Reduce D capacity**: halve the channel multipliers (512 → 256).

## Skill 4: Switch to WGAN-GP

Replace BCELoss with Wasserstein loss + gradient penalty:

```python
# src/train.py — replace criterion
critic_real = D(real).mean()
critic_fake = D(fake).mean()
gp = ((torch.autograd.grad(critic_real, real, create_graph=True)[0])**2).sum(1).mean()
d_loss = -(critic_real - critic_fake) + 10.0 * gp
```

WGAN-GP is more stable than DCGAN but slower per step.

## Skill 5: Add spectral normalization

```python
# src/model.py
import torch.nn.utils.spectral_norm as spectral_norm
self.conv1 = spectral_norm(nn.Conv2d(...))
```

Spectral norm on D is a lightweight alternative to WGAN-GP that constrains
Lipschitz constant.

## Skill 6: Convert to a conditional GAN

Add an embedding for class label:
```python
# Generator input: (z, y) → concat or AdaIN
class_cond = self.embed(y).unsqueeze(-1).unsqueeze(-1)
x = torch.cat([z, class_cond], dim=1)
```

Requires class labels in the dataset (CelebA has ~40 attributes — pick one).

## Pitfalls
- **Strided convs** (not pooling) in both G and D.
- **LeakyReLU(0.2)** in D, **ReLU** in G.
- **Tanh output** in G (match data range [-1, 1]).
- **`nn.BatchNorm2d`** in both G and D, **except** the last layer of G
  and first layer of D.
- **`torch.zeros_like`** for fake labels (not `torch.ones`).

