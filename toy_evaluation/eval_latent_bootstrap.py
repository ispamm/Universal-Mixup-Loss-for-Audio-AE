#!/usr/bin/env python3
"""
Latent-space metrics with bootstrap uncertainty (convexity, effective rank).

The eval battery reports `latent_convexity` and `latent_effective_rank` as single
pool-level numbers with no uncertainty. This script recomputes them on the SAME
256-clip pool the battery uses and adds a bootstrap standard error:

  * effective rank  -- resample the 256 clips with replacement (B times) and
    recompute the participation ratio of the centred pooled codes;
  * convexity       -- draw 512 pairs (in-batch), take the constant-power
    midpoint, measure its nearest-neighbour distance to the pool excluding the
    pair's own endpoints, and bootstrap over pairs.

Usage:
    python toy_evaluation/eval_latent_bootstrap.py --checkpoints ckpt1 ... \
        --num_clips 256 --num_mixups 512 --boot 500 --device cuda:0
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper
from toy_evaluation.latent_topology_toy import trig_mix, _compute_convexity_batched


def eff_rank(Z):
    """Participation-ratio effective rank of a (N, P) centred pool."""
    Z = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", nargs="+",
                   default=["/data/wav48_silence_trimmed",
                            "/data/fsd50k/FSD50K.dev_audio",
                            "/data/fma_small"])
    p.add_argument("--num_clips", type=int, default=256)
    p.add_argument("--num_mixups", type=int, default=512)
    p.add_argument("--num_recon_clips", type=int, default=2000,
                   help="Battery draws the recon indices BEFORE the topo indices "
                        "from the same RNG stream; replicate that to get the "
                        "identical clip pool.")
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--clip_seconds", type=float, default=2.0)
    p.add_argument("--output_dir", default="toy_evaluation/eval_exp1_metrics")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    n = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    stems = collect_validation_stems(args.data_dirs, sample_rate=sr)
    # EXACT replication of evaluate_all.py's draw order: one Random(seed) is
    # created, the recon indices are sampled first, and the topo indices are the
    # SECOND sample from that same stream.
    rng = random.Random(args.seed)
    _recon = rng.sample(range(len(stems)), min(args.num_recon_clips, len(stems)))
    clip_indices = rng.sample(range(len(stems)), min(args.num_clips, len(stems)))
    print(f"[Setup] {len(clip_indices)} clips (battery-identical pool: recon drawn "
          f"first from the same RNG stream)", flush=True)

    X = torch.stack([torch.from_numpy(
        load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy())
        for i in clip_indices]).to(device)

    # mixup pair indices (deterministic), independent of the bootstrap resample
    prng = np.random.default_rng(args.seed)
    pi = prng.integers(0, len(clip_indices), size=args.num_mixups)
    pj = prng.integers(0, len(clip_indices), size=args.num_mixups)
    same = pi == pj
    pj[same] = (pj[same] + 1) % len(clip_indices)
    pi_t = torch.from_numpy(pi).to(device)
    pj_t = torch.from_numpy(pj).to(device)

    brng = np.random.default_rng(args.seed + 1)
    results = {}
    for ci, ck in enumerate(args.checkpoints):
        name = Path(ck).stem
        print(f"[{ci+1}/{len(args.checkpoints)}] {name}", flush=True)
        ae = ToyModelWrapper(ck, device)
        with torch.no_grad():
            Z = []
            for s in range(0, len(X), 32):
                z = ae.model.encode(X[s:s+32].unsqueeze(1))
                if ae.model_type == "transformer":
                    z = z.permute(0, 2, 1)
                Z.append(z.flatten(1))
            Z = torch.cat(Z, 0).cpu().numpy()        # (N, P)

        r_point = eff_rank(Z)
        boot_r = []
        for _ in range(args.boot):
            idx = brng.integers(0, len(Z), size=len(Z))
            boot_r.append(eff_rank(Z[idx]))
        r_std = float(np.std(boot_r, ddof=1))

        # ---- convexity (trig midpoint), bootstrapped over pairs ----
        Zt = torch.from_numpy(Z).to(device)
        Zi = Zt[pi_t]; Zj = Zt[pj_t]
        Z_mid = trig_mix(Zi.unsqueeze(1) if Zi.dim() == 2 else Zi,
                         Zj.unsqueeze(1) if Zj.dim() == 2 else Zj, 0.5)
        if Z_mid.dim() == 3:
            Z_mid = Z_mid.squeeze(1)
        # per-pair nearest-neighbour distance to the pool (excluding own endpoints)
        dists = []
        for s in range(0, len(Z_mid), 64):
            chunk = Z_mid[s:s+64]
            d = torch.cdist(chunk, Zt)                     # (m, N)
            rows = torch.arange(chunk.shape[0], device=device)
            gi = pi_t[s:s+64]; gj = pj_t[s:s+64]
            d[rows, gi] = float("inf")
            d[rows, gj] = float("inf")
            dists.append(d.min(dim=1).values.cpu().numpy())
        dists = np.concatenate(dists)
        c_point = float(dists.mean())
        boot_c = [float(brng.choice(dists, size=len(dists), replace=True).mean())
                  for _ in range(args.boot)]
        c_std = float(np.std(boot_c, ddof=1))

        results[name] = {
            "effective_rank": r_point, "effective_rank_boot_std": r_std,
            "convexity_trig": c_point, "convexity_boot_std": c_std,
            "n_clips": int(len(Z)), "n_mixups": int(len(dists)),
            "boot": args.boot,
        }
        print(f"   eff_rank {r_point:.2f} +- {r_std:.2f} | "
              f"convexity {c_point:.2f} +- {c_std:.2f}", flush=True)
        del ae, Zt, Zi, Zj, Z_mid
        torch.cuda.empty_cache()
        (out / "exp1_latent_bootstrap.json").write_text(json.dumps(results, indent=2))

    print(f"\nSaved: {out / 'exp1_latent_bootstrap.json'}")


if __name__ == "__main__":
    main()
