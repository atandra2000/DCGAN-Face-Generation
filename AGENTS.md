# AGENTS.md — DCGAN-Face-Generation

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific rules only.

> **Project:** `Vision/DCGAN-Face-Generation/` · **Type:** unconditional GAN
> **Task:** 64×64 face generation from Gaussian noise
> **Hardware:** 2× NVIDIA T4 (Kaggle) · **Status:** complete (50 epochs)
> **Architecture detail:** see `README.md §9`; cross-project guidance at
> `.agents/skills/generative-vision-gans-vaes/SKILL.md §1`.

## 1. Subagent: `gan-trainer-debugger`

**Triggers:** "DCGAN mode collapse", "Generator produces noise",
"Wasserstein loss not decreasing", "GAN divergence after epoch 10",
"D and G loss oscillating."

**Knows cold:**
- DCGAN (Radford et al., 2015) implemented from scratch.
- **Generator:** 5 ConvTranspose2d (100 → 512 → 256 → 128 → 64 → 3),
  BatchNorm + ReLU, Tanh output (~3.6M params).
- **Discriminator:** 5 Conv2d (3 → 64 → 128 → 256 → 512 → 1), BatchNorm +
  LeakyReLU(0.2), Sigmoid (~2.8M params).
- Weight init N(0, 0.02); strided convs (no pooling).
- Training: **non-saturating** GAN loss (maximize log D(G(z))), BCELoss,
  Adam (lr 2e-4, β1=0.5), 50 epochs, batch 128. Comet ML tracking.
- CelebA (202,599 images), 64×64, normalized [-1, 1], HFlip/rotation/
  color-jitter augmentation.
- Results: G loss 5.5 → 1.82; D loss → ln(2) ≈ 0.693; D(G(z)) 0.04 → 0.44
  (post-softmax probability that D classifies G(z) as real).

## 2. Hard rules

1. **Never** use the saturating GAN loss form
   (`F.binary_cross_entropy_with_logits(D_out, real_label)` for the
   generator). Always use the **non-saturating** form.
2. **Never** disable BatchNorm in G or D. Without it, training diverges.
3. **Always** use `lr=2e-4, betas=(0.5, 0.999)` (the standard DCGAN recipe).
4. **Always** use N(0, 0.02) weight init (the original Radford recipe).
   Other inits cause mode collapse.

## 3. Mode-collapse diagnostic

| Symptom | Cause | Fix |
|---------|-------|-----|
| G produces same image | Mode collapse | Lower lr, add noise to D inputs |
| G loss → 0, D loss → 0 | G fools D too easily | Label smoothing (0.9 not 1.0) |
| G loss → ∞, D loss → 0 | G exploding | Clip gradients, lower lr |
| D loss → 0, G loss stable | D too strong | Add noise to D, fewer D steps |
| Both losses flat | Vanishing gradient | LeakyReLU not ReLU in D |

## 4. Cross-references

- For label-smoothed BCE in WGAN-GP style: see
  `Vision/FaceAgingCycleGAN/AGENTS.md`.
- For β-VAE posterior collapse: see `Vision/FaceGenerationVAE/AGENTS.md`.
- StableDiffusion DDIM clamp bug: `Vision/StableDiffusion/AGENTS.md §2`.

## 5. Files

- `configs/config.py` — `DCGANConfig`.
- `src/{model,dataset,train,generate,utils}.py`.
- `assets/training_curves.png`.
