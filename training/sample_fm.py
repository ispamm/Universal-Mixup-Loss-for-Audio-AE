#!/usr/bin/env python3
"""
Sample audio from a trained flow-matching DiT in the AE latent space.

Generates audio in three ways:
  1. UNCONDITIONAL: integrate the learned velocity field from Gaussian noise
     to a data latent, then decode with the AE.
  2. MIX (latent arithmetic): generate two samples, form
     z_mix = cos(theta)*z_A + sin(theta)*z_B (the cone operation the AE was
     trained on), decode. On an additive latent this is a valid audio morph.
  3. INTERPOLATION: generate two samples and sweep theta in [0, pi/2],
     decoding each step -> a morphing audio transition.

Rectified-flow sampling (matches the training convention):
    training:  z_t = (1-t) z_data + t z_noise,   v_target = z_noise - z_data
    hence      dz/dt = v  and  t=0 is data, t=1 is noise.
    -> sample by integrating t: 1 -> 0  with  z <- z + v(z,t) * dt,  dt < 0.

Usage:
    python training/sample_fm.py \
        --fm_checkpoint training/checkpoints/fm_pilot/fm_univ_best.pt \
        --num_samples 4 --steps 100 \
        --output_dir evaluation/audio_demo
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.model_utils import AutoencoderWrapper


# ---- DiT definition (must match train_fm_pilot.py exactly) ----
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class LatentDiT(nn.Module):
    def __init__(self, channels=64, embed_dim=384, depth=6, num_heads=8,
                 mlp_ratio=4.0, t_dim=256):
        super().__init__()
        self.channels = channels
        self.in_proj = nn.Linear(channels, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
            nn.SiLU(), nn.Linear(t_dim, embed_dim))
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, channels)

    def forward(self, z, t):
        x = z.permute(0, 2, 1)
        x = self.in_proj(x)
        c = self.time_embed(t.unsqueeze(-1))
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.out_proj(x)
        return x.permute(0, 2, 1)


@torch.no_grad()
def sample_latents(model, shape, steps, device, seed=None, method="euler"):
    """Integrate rectified flow from noise (t=1) to data (t=0)."""
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(shape, device=device)          # t = 1 (noise)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]                     # negative
        t_batch = torch.full((shape[0],), float(t), device=device)
        v = model(z, t_batch)
        if method == "euler":
            z = z + v * dt
        elif method == "midpoint":                 # 2nd-order (rk2)
            z_mid = z + v * (dt / 2)
            t_mid = torch.full((shape[0],), float(t + dt / 2), device=device)
            v_mid = model(z_mid, t_mid)
            z = z + v_mid * dt
    return z                                        # t = 0 (data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--ae_checkpoint", default=None,
                   help="Override AE (default: the one stored in the FM checkpoint)")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="euler", choices=["euler", "midpoint"])
    p.add_argument("--output_dir", default="evaluation/audio_demo")
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.fm_checkpoint, map_location="cpu")
    ae_ckpt = args.ae_checkpoint or ck["ae_checkpoint"]
    print(f"[FM]  {args.fm_checkpoint} (epoch {ck.get('epoch')}, val {ck.get('val_loss'):.4f})")
    print(f"[AE]  {ae_ckpt}")

    model = LatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                      depth=ck["depth"], num_heads=ck["num_heads"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    ae = AutoencoderWrapper(ae_ckpt, device)
    clip_samples = int(args.sample_rate * args.clip_seconds)
    shape = (args.num_samples, ck["channels"], ck["t_lat"])
    print(f"[Gen] {args.num_samples} samples, shape {shape}, {args.steps} steps, {args.method}")

    # ---- (1) unconditional samples ----
    z = sample_latents(model, shape, args.steps, device, seed=args.seed, method=args.method)
    print(f"[Gen] latent stats: mean={z.mean():.4f} std={z.std():.4f}")
    for i in range(args.num_samples):
        audio = ae.decode(z[i:i+1], clip_samples)
        peak = np.abs(audio).max()
        path = out / f"fm_uncond_{i:02d}.wav"
        sf.write(path, audio, args.sample_rate)
        print(f"  wrote {path.name}  (peak {peak:.3f})")

    # ---- (2) cone mix of two generated samples (latent arithmetic) ----
    if args.num_samples >= 2:
        for (i, j) in [(0, 1)]:
            for theta_deg in [30, 45, 60]:
                th = math.radians(theta_deg)
                z_mix = math.cos(th) * z[i:i+1] + math.sin(th) * z[j:j+1]
                audio = ae.decode(z_mix, clip_samples)
                path = out / f"fm_conemix_{i}{j}_theta{theta_deg:02d}.wav"
                sf.write(path, audio, args.sample_rate)
                print(f"  wrote {path.name}")

    # ---- (3) full interpolation sweep between two generated samples ----
    if args.num_samples >= 2:
        for k, alpha in enumerate(np.linspace(0, 1, 11)):
            th = alpha * math.pi / 2
            z_mix = math.cos(th) * z[0:1] + math.sin(th) * z[1:2]
            audio = ae.decode(z_mix, clip_samples)
            sf.write(out / f"fm_interp_{k:02d}_alpha{alpha:.1f}.wav", audio, args.sample_rate)
        print(f"  wrote 11-step interpolation sweep (fm_interp_*.wav)")

    # ---- also a single long concatenated demo for easy listening ----
    parts = []
    for i in range(args.num_samples):
        parts.append(ae.decode(z[i:i+1], clip_samples))
        parts.append(np.zeros(int(0.3 * args.sample_rate), dtype=np.float32))
    concat = np.concatenate(parts)
    sf.write(out / "fm_uncond_all.wav", concat, args.sample_rate)
    print(f"  wrote fm_uncond_all.wav (all samples concatenated)")

    meta = {
        "fm_checkpoint": args.fm_checkpoint,
        "ae_checkpoint": ae_ckpt,
        "epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"),
        "num_samples": args.num_samples, "steps": args.steps,
        "method": args.method, "seed": args.seed,
        "sample_rate": args.sample_rate, "clip_seconds": args.clip_seconds,
    }
    (out / "fm_sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAll audio in: {out}")


if __name__ == "__main__":
    main()
