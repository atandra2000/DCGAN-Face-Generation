# AGENTS.md — DCGAN-Face-Generation

> **Project:** `Vision/DCGAN-Face-Generation/` · **Type:** unconditional GAN
> **Task:** 64×64 face generation from Gaussian noise
> **Hardware:** 2× NVIDIA T4 (Kaggle) · **Status:** complete (50 epochs)

From-scratch PyTorch implementation of DCGAN (Radford et al., 2015) that
synthesizes photorealistic 64×64 celebrity faces from 100-dim Gaussian noise,
trained adversarially with no labels.

---

## 1. Subagent: `gan-trainer-debugger`

**Trigger:** "DCGAN mode collapse", "Generator produces noise", "Wasserstein
loss not decreasing", "GAN divergence after epoch 10", "D and G loss
oscillating."

**System prompt:**
You are a senior GAN training engineer. You know the adversarial training
instability patterns cold and have shipped three GAN projects in this
portfolio (DCGAN, FaceAgingCycleGAN, FaceGenerationVAE). When in doubt,
reference `assets/training_curves.png` for the canonical DCGAN result.

**Architecture (DCGAN, Radford et al. 2015):**
- **Generator:** 5 ConvTranspose2d (100 → 512 → 256 → 128 → 64 → 3),
  BatchNorm + ReLU, Tanh output (~3.6M params).
- **Discriminator:** 5 Conv2d (3 → 64 → 128 → 256 → 512 → 1),
  BatchNorm + LeakyReLU(0.2), Sigmoid (~2.8M params).
- Weight init N(0, 0.02); strided convs (no pooling).

**Training:**
- Non-saturating GAN loss (maximize log D(G(z)), not minimize log(1−D(G(z)))).
- BCELoss, Adam (lr 2e-4, β1=0.5), 50 epochs, batch 128.
- Comet ML tracking.

**Dataset:** CelebA (202,599 images), 64×64, normalized [-1,1],
HFlip/rotation/color-jitter augmentation.

**Results:**
- G loss 5.5 → 1.82.
- D loss → ln(2) ≈ 0.693.
- D(G(z)) 0.04 → 0.44 (post-softmax probability that D classifies G(z) as real).

**Files:**
- `configs/config.py` — `DCGANConfig`.
- `src/{model,dataset,train,generate,utils}.py`.
- `assets/training_curves.png`.

**Hard rules:**
1. **Never** use BCE-with-logits as the loss in the buggy "saturating"
   form (`F.binary_cross_entropy_with_logits(D_out, real_label)` for
   *generator*). Always use the **non-saturating** form
   (`-D_fake.log()` equivalent).
2. **Never** disable BatchNorm in G or D. Without it, training diverges.
3. **Always** use `lr=2e-4, betas=(0.5, 0.999)` (the standard DCGAN recipe).
4. **Always** use N(0, 0.02) init (the original Radford recipe). Other
   inits cause mode collapse.

**Mode-collapse diagnostic:**
| Symptom | Cause | Fix |
|---------|-------|-----|
| G produces same image | Mode collapse | Lower lr, add noise to D inputs |
| G loss → 0, D loss → 0 | G fools D too easily | Add label smoothing (0.9 not 1.0) |
| G loss → ∞, D loss → 0 | G exploding | Clip gradients, lower lr |
| D loss → 0, G loss stable | D too strong | Add noise to D, fewer D steps |
| Both losses flat | Vanishing gradient | Use LeakyReLU not ReLU in D |

**Cross-references:**
- For label-smoothed BCE in WGAN-GP style: see
  `Vision/FaceAgingCycleGAN/AGENTS.md`.
- For β-VAE posterior collapse: see `Vision/FaceGenerationVAE/AGENTS.md`.

