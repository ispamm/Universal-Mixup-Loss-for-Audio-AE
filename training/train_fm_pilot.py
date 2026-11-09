#!/usr/bin/env python3
"""
Flow-Matching (Rectified Flow) pilot on AE latents — DiT architecture.

Trains a small Diffusion Transformer (DiT) on the latents of a frozen
autoencoder, using the SAME rectified-flow recipe:
  - Linear interpolant: z_t = (1-t) * z_data + t * noise
  - Velocity target:    v = noise - data
  - Minibatch OT coupling (Sinkhorn, GPU-only) — straightens transport paths
  - Timestep conditioning via adaLN (adaptive layer norm, as in DiT/Peebles 2023)

Two runs are compared:
  - FM on universal-AE latents (cone-structured, FM-compatible)
  - FM on softnorm-AE latents (no additivity, baseline)

The metric is the RF loss convergence speed + floor. If the cone-structured
latent gives straighter OT-coupled paths (M-FM1), the universal model should
converge faster and to a lower loss.

==================  Design choices (explained)  ==================

WHY A DiT (not a U-Net):
  SAME itself uses a Diffusion Transformer (stable_audio_3/models/dit.py).
  A DiT is the architecturally correct choice for latent diffusion: it
  treats the latent as a sequence of tokens (T_lat, D) and applies
  self-attention, which naturally handles the 2D latent grid
  (timesteps × channels). A 1D U-Net is a convolutional shortcut that
  doesn't match the paper's framing. Using a DiT makes the pilot a faithful
  miniature of the SAME latent-diffusion pipeline, so the comparison is
  directly relevant to the paper's FM claim.

WHY TRAINING STEMS (not validation):
  The FM model must learn the *training* latent distribution, not the
  held-out validation distribution. The AE was trained on 90% of the tracks;
  its latents on those tracks represent the full training distribution the
  FM model will see in production. Using only validation clips (10%) would
  under-sample the distribution and make the convergence comparison
  unreliable. We hold out 10% of the *training* stems as the FM validation
  set (to monitor overfitting), but the FM training set is the full 90%
  training split of the AE.

WHY 16 kHz:
  Both AEs (universal and softnorm) were trained at 16 kHz. The FM comparison
  is apples-to-apples at 16 kHz — the absolute audio quality doesn't matter,
  only the *relative* difference in how easily the FM model learns each
  latent. Training a 44.1 kHz softnorm AE just for the baseline is
  unnecessary GPU cost.

WHY 10-SECOND CLIPS:
  10s is the standard evaluation clip length. Using 10s (not 2s) gives the
  FM model longer temporal context, which is what it will see in production.
  The latent at 10s / 16 kHz is (64, 625) — 625 temporal tokens × 64
  channels — a reasonable sequence length for a small DiT.

WHY OT COUPLING:
  SAME's RF training uses minibatch optimal transport (Sinkhorn) to match
  noise to data, which straightens transport paths and speeds convergence
  (MelodyFlow, arXiv:2407.03648). This is the mechanism our M-FM1 theory
  predicts should benefit from the cone-structured latent: if the data
  manifold is locally convex (the cone property), the OT-matched paths
  are straighter, so the velocity field has lower variance and the RF loss
  is smaller at convergence.

WHY THIS DiT SIZE:
  6 layers, 8 heads, embed_dim=384, ~10M params. Small enough to train in
  ~30 min on one L40, large enough to show meaningful convergence behaviour.
  This is a pilot to validate M-FM1, not a production model.

WHY THE SAME RF RECIPE AS SAME:
  We replicate the exact RF objective from stable_audio_3/training/diffusion.py:
  alpha = 1-t, sigma = t, z_t = alpha*data + sigma*noise, target = noise - data.
  This ensures the pilot is directly comparable to the production pipeline.

Usage:
    python training/train_fm_pilot.py \
        --ae_checkpoint training/checkpoints/transformer_univ_warm_v2_best.pt \
        --run_name fm_univ --device cuda:3
"""
from __future__ import annotations
import argparse, math, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data import collect_validation_stems, load_stem_clip, get_cache_manifest, get_validation_tracks
from evaluation.model_utils import AutoencoderWrapper


# ---- collect TRAINING stems (the 90% the AE was trained on) ----
def collect_training_stems(data_dirs=None, cache_dir=None, sample_rate=16000):
    """Return (pt_path, stem_name, track_id) for every TRAINING stem
    (the complement of collect_validation_stems)."""
    manifest = get_cache_manifest(data_dirs, cache_dir, sample_rate=sample_rate)
    val_tracks = get_validation_tracks(manifest)
    stems = []
    for track_id, paths in manifest["tracks"].items():
        if track_id in val_tracks:
            continue  # skip validation tracks
        for p in paths:
            stem_name = os.path.splitext(os.path.basename(p))[0]
            stems.append((p, stem_name, track_id))
    stems.sort()
    return stems


import os  # needed by collect_training_stems

# Force unbuffered stdout so we can track progress in real-time
import functools
print = functools.partial(print, flush=True)


# =============================================================================
#  Minimal Diffusion Transformer (DiT) for latent velocity prediction
#  Architecture: patchify → DiT blocks (adaLN-Zero) → unpatchify
#  Based on Peebles & Xie (2023), adapted for 1D latent sequences.
# =============================================================================

class DiTBlock(nn.Module):
    """A DiT block with adaLN-Zero conditioning (Peebles & Xie 2023)."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        # adaLN-Zero: scale, shift, gate (init gates to 0 = identity at start)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        # c: (B, dim) timestep embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self._attn(
            self._mod(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self._mod(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def _attn(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def _mod(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LatentDiT(nn.Module):
    """Diffusion Transformer operating on (B, D, T_lat) latents.

    The latent is treated as a sequence of T_lat tokens, each of dimension D.
    Timestep conditioning is injected via adaLN-Zero in every block.
    """
    def __init__(self, channels=64, embed_dim=384, depth=6, num_heads=8,
                 mlp_ratio=4.0, t_dim=256):
        super().__init__()
        self.channels = channels
        # input projection: D → embed_dim
        self.in_proj = nn.Linear(channels, embed_dim)
        # timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
            nn.SiLU(), nn.Linear(t_dim, embed_dim))
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        # output: adaLN-Zero final layer + projection back to D
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.out_proj = nn.Linear(embed_dim, channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, t):
        # z: (B, D, T_lat) → (B, T_lat, D) for attention
        x = z.permute(0, 2, 1)
        x = self.in_proj(x)  # (B, T_lat, embed_dim)
        c = self.time_embed(t.unsqueeze(-1))  # (B, embed_dim)
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.out_proj(x)  # (B, T_lat, D)
        return x.permute(0, 2, 1)  # (B, D, T_lat)


# =============================================================================
#  Minibatch OT coupling (Sinkhorn) — same as stable_audio_3/training/diffusion.py
# =============================================================================

def ot_coupling(data, noise):
    """Minibatch OT coupling via Sinkhorn (GPU-only, no CPU sync).
    Replicates stable_audio_3/training/diffusion.py:343-369."""
    B = data.shape[0]
    if B <= 1:
        return noise
    with torch.no_grad():
        d_flat = data.reshape(B, -1)
        n_flat = noise.reshape(B, -1)
        aa = (d_flat * d_flat).sum(dim=1, keepdim=True)
        bb = (n_flat * n_flat).sum(dim=1, keepdim=True)
        cost = aa + bb.T - 2.0 * (d_flat @ n_flat.T)
        log_P = -cost / cost.detach().mean()
        for _ in range(20):
            log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
        P = log_P.exp()
        col_indices = torch.empty(B, dtype=torch.long, device=P.device)
        used = torch.zeros(B, dtype=torch.bool, device=P.device)
        for i in range(B):
            P[i, used] = -1
            col_indices[i] = P[i].argmax()
            used[col_indices[i]] = True
    return noise[col_indices]


# =============================================================================
#  Main training loop
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of training stems to hold out for FM validation")
    p.add_argument("--max_train_clips", type=int, default=20000,
                   help="Subsample training clips to this many (pilot speed; 0 = all)")
    p.add_argument("--use_ot_coupling", action="store_true", default=True)
    p.add_argument("--no_ot_coupling", dest="use_ot_coupling", action="store_false")
    p.add_argument("--output_dir", default="training/checkpoints/fm_pilot")
    p.add_argument("--wandb_project", default="umix_ae_phase",
                   help="WandB project name (set to empty string to disable)")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    clip_samples = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- WandB ----
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "ae_checkpoint": args.ae_checkpoint,
                "sample_rate": sr,
                "clip_seconds": args.clip_seconds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "use_ot_coupling": args.use_ot_coupling,
                "max_train_clips": args.max_train_clips,
            },
        )

    # ---- Load frozen AE ----
    print(f"[Setup] Loading AE: {args.ae_checkpoint}")
    ae = AutoencoderWrapper(args.ae_checkpoint, device)

    # ---- Collect TRAINING stems (the 90% the AE was trained on) ----
    print(f"[Setup] Collecting TRAINING stems (sr={sr})")
    all_stems = collect_training_stems(sample_rate=sr)
    print(f"[Setup] {len(all_stems)} training stems available")

    # Hold out a fraction for FM validation
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_stems))
    n_val = int(len(all_stems) * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_stems = [all_stems[i] for i in train_idx]
    val_stems = [all_stems[i] for i in val_idx]

    # Subsample training clips for pilot speed
    if args.max_train_clips > 0 and len(train_stems) > args.max_train_clips:
        train_stems = train_stems[:args.max_train_clips]
        print(f"[Setup] Subsampled to {len(train_stems)} train clips (pilot)")
    # Also subsample val to keep it proportional
    max_val = min(len(val_stems), max(1000, len(train_stems) // 10))
    val_stems = val_stems[:max_val]

    print(f"[Setup] FM train: {len(train_stems)}, FM val: {len(val_stems)}")

    # ---- Pre-encode all clips into latents ----
    print(f"[Setup] Pre-encoding train clips ({args.clip_seconds}s)...")
    t0 = time.time()
    train_z = encode_dataset(ae, train_stems, clip_samples, device, "train")
    print(f"[Setup] Pre-encoding val clips...")
    val_z = encode_dataset(ae, val_stems, clip_samples, device, "val")
    print(f"[Setup] Encoding done in {time.time()-t0:.0f}s")
    print(f"[Setup] train latents: {train_z.shape}, val latents: {val_z.shape}")
    print(f"[Setup] latent stats: mean={train_z.mean():.4f}, std={train_z.std():.4f}, var={train_z.var():.4f}")

    # ---- FM model (DiT) ----
    D, T_lat = train_z.shape[1], train_z.shape[2]
    model = LatentDiT(
        channels=D, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=4.0, t_dim=256
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Setup] DiT: {n_params/1e6:.1f}M params, latent ({D}, {T_lat}), "
          f"depth={args.depth}, dim={args.embed_dim}, heads={args.num_heads}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_z) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    # Paths for saved weights (best + final), so samples can be generated later
    best_ckpt = out / f"{args.run_name}_best.pt"
    final_ckpt = out / f"{args.run_name}_final.pt"

    # ---- Training loop ----
    log = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_z))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(train_z), args.batch_size):
            idx = perm[i:i+args.batch_size]
            z_data = train_z[idx].to(device)
            B = z_data.shape[0]
            # sample timesteps uniform in [0, 1]
            t = torch.rand(B, device=device)
            # noise
            noise = torch.randn_like(z_data)
            # OT coupling
            if args.use_ot_coupling and B > 1:
                noise = ot_coupling(z_data, noise)
            # RF interpolant: z_t = (1-t)*data + t*noise  (alpha=1-t, sigma=t)
            alpha = (1 - t).view(B, 1, 1)
            sigma = t.view(B, 1, 1)
            z_t = alpha * z_data + sigma * noise
            # velocity target: v = noise - data
            v_target = noise - z_data
            # predict velocity
            v_pred = model(z_t, t)
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            # Per-batch wandb logging (every 50 steps)
            if wandb_run and (n_batches % 50 == 0):
                wandb_run.log({
                    "fm/batch_loss": loss.item(),
                    "fm/batch_lr": scheduler.get_last_lr()[0],
                    "step": epoch * steps_per_epoch + n_batches,
                })

        avg_train = epoch_loss / n_batches

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_loss = 0
            n_val = 0
            for i in range(0, len(val_z), args.batch_size):
                z_data = val_z[i:i+args.batch_size].to(device)
                B = z_data.shape[0]
                t = torch.rand(B, device=device)
                noise = torch.randn_like(z_data)
                if args.use_ot_coupling and B > 1:
                    noise = ot_coupling(z_data, noise)
                alpha = (1 - t).view(B, 1, 1)
                sigma = t.view(B, 1, 1)
                z_t = alpha * z_data + sigma * noise
                v_target = noise - z_data
                v_pred = model(z_t, t)
                val_loss += F.mse_loss(v_pred, v_target).item()
                n_val += 1
            avg_val = val_loss / n_val

        log.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})
        marker = " *" if avg_val < best_val else ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "channels": D, "t_lat": T_lat,
                "embed_dim": args.embed_dim, "depth": args.depth,
                "num_heads": args.num_heads,
                "ae_checkpoint": args.ae_checkpoint,
                "epoch": epoch + 1, "val_loss": avg_val,
            }, best_ckpt)
        if wandb_run:
            wandb_run.log({
                "fm/train_loss": avg_train,
                "fm/val_loss": avg_val,
                "fm/best_val": best_val,
                "fm/lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            })
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}/{args.epochs}] train={avg_train:.6f} val={avg_val:.6f}{marker}")

    # ---- Save ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "channels": D, "t_lat": T_lat,
        "embed_dim": args.embed_dim, "depth": args.depth,
        "num_heads": args.num_heads,
        "ae_checkpoint": args.ae_checkpoint,
        "epoch": args.epochs, "val_loss": avg_val,
    }, final_ckpt)
    result = {
        "run_name": args.run_name,
        "ae_checkpoint": args.ae_checkpoint,
        "latent_stats": {
            "mean": float(train_z.mean()),
            "std": float(train_z.std()),
            "var": float(train_z.var()),
        },
        "n_train": len(train_z), "n_val": len(val_z),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "dit_config": {"embed_dim": args.embed_dim, "depth": args.depth,
                       "num_heads": args.num_heads, "n_params_M": n_params / 1e6},
        "use_ot_coupling": args.use_ot_coupling,
        "loss_curve": log,
        "best_val": best_val,
    }
    out_file = out / f"{args.run_name}_fm_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved weights: {best_ckpt}")
    print(f"Saved weights: {final_ckpt}")
    print(f"Saved: {out_file}")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Latent var: {train_z.var():.4f} (ideal ~1.0)")
    if wandb_run:
        wandb_run.finish()


def encode_dataset(ae, stem_list, clip_samples, device, label=""):
    """Encode a list of stems into latents."""
    latents = []
    for i, (pt_path, _, _) in enumerate(stem_list):
        audio = load_stem_clip(pt_path, clip_samples, centre=True).squeeze().numpy()
        z = ae.encode(audio)  # (1, D, T_lat)
        latents.append(z.squeeze(0).cpu())  # (D, T_lat)
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] encoded {i+1}/{len(stem_list)}")
    return torch.stack(latents)


if __name__ == "__main__":
    main()
