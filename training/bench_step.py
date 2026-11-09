#!/usr/bin/env python3
"""Estimate per-batch / per-epoch training cost on synthetic data.

Replicates one training iteration (discriminator step + generator step with
SAME loss, adversarial loss, soft-norm KL, and trig latent mixup) closely
enough to project per-epoch wall time.  Does NOT touch the dataset.
"""
import math
import sys
import time

import torch

sys.path.insert(0, ".")

from model.discriminator import STFTDiscriminator
from model.transformer_autoencoder import TransformerAutoencoder
from training.losses import SAMELoss

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

torch.manual_seed(0)
device = torch.device(DEV)

model = TransformerAutoencoder(
    latent_dim=64, d_model=512, nhead=8, num_layers=6,
    dim_feedforward=2048, dropout=0.0, patch_size=256,
    activation="leaky_relu", decoder_num_snake_layers=2,
    encoder_final_norm=False, decoder_final_norm=False,
    use_soft_norm=True,
).to(device)

disc = STFTDiscriminator(n_fft=1024, hop_length=256, hidden_channels=32).to(device)
loss_fn = SAMELoss(eps=1e-8).to(device)
mse = torch.nn.MSELoss().to(device)

opt_G = torch.optim.AdamW(model.parameters(), lr=1e-4)
opt_D = torch.optim.AdamW(disc.parameters(), lr=1e-4)

audio = torch.randn(BS, 1, 32000, device=device)


def step():
    z = model.encode(audio)
    recon = model.decode(z, original_length=audio.shape[-1])

    # discriminator
    _f, d_real = disc(audio)
    _f, d_fake = disc(recon.detach())
    loss_D = 0.5 * mse(d_real, torch.ones_like(d_real)) + 0.5 * mse(d_fake, torch.zeros_like(d_fake))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # generator: SAME + adv + KL + mixup
    loss_same, mag = loss_fn(recon, audio, use_phase=False, return_components=True)
    _f, d_g = disc(recon)
    loss_adv = mse(d_g, torch.ones_like(d_g))
    total = loss_same + 0.1 * loss_adv
    if model.soft_norm is not None and model.soft_norm.z_norm_clean is not None:
        zc = model.soft_norm.z_norm_clean
        B, L, D = zc.shape
        mu_t = zc.mean(1); s_t = zc.var(1, unbiased=False)
        kl_t = (mu_t**2 + s_t - torch.log(s_t + 1e-8) - 1).mean()
        mu_c = zc.mean(2); s_c = zc.var(2, unbiased=False)
        kl_c = (mu_c**2 + s_c - torch.log(s_c + 1e-8) - 1).mean()
        total = total + 1.0 * (kl_t + 0.4 * kl_c)

    alpha = torch.rand(BS, 1, 1, device=device)
    a_roll = torch.roll(audio, 1, 0)
    z_roll = torch.roll(z, 1, 0)
    th = alpha * (math.pi / 2)
    z_mix = torch.cos(th) * z + torch.sin(th) * z_roll
    a_mix = torch.cos(th) * audio + torch.sin(th) * a_roll
    recon_mix = model.decode(z_mix, original_length=audio.shape[-1])
    loss_mix = loss_fn(recon_mix, a_mix, use_phase=False)
    total = total + 1.0 * loss_mix

    opt_G.zero_grad(); total.backward(); opt_G.step()


# warmup
for _ in range(2):
    step()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    step()
torch.cuda.synchronize()
per_batch = (time.time() - t0) / N

TRAIN_STEMS = 123000  # ~90% of 137294 files
batches_per_epoch = TRAIN_STEMS / BS
per_epoch_min = per_batch * batches_per_epoch / 60.0

print(f"device={DEV} batch_size={BS}")
print(f"per-batch training step: {per_batch*1000:.0f} ms")
print(f"batches/epoch (train stems ~{TRAIN_STEMS}): {batches_per_epoch:.0f}")
print(f"PROJECTED per-epoch time: {per_epoch_min:.1f} min")
print(f"PROJECTED 50-epoch trial: {per_epoch_min*50/60:.1f} h")
print(f"PROJECTED 200-epoch trial: {per_epoch_min*200/60:.1f} h")
