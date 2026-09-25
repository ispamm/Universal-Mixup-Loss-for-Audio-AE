#!/usr/bin/env python3
"""
E2 — direct measurement of the learned trajectory's straightness.

Why this exists.  The step x estimator grid shows univ's quality is flat from 4
to 32 Euler steps while softnorm needs 8-16, which *implies* univ's learned field
has low curvature.  This script measures curvature directly instead of inferring
it, so M-FM1 can be stated rather than argued, and it produces the trajectory
straightness numbers behind the paper's figure.

Definitions (all per sample, then averaged):

    S  = ||z(t=0) - z(t=1)|| / integral_0^1 ||v(z(t), t)|| dt
         the standard rectified-flow straightness statistic; exactly 1.0 for a
         straight path traversed at constant speed, < 1 for a curved one.

    dev(t) = || z(t) - chord(t) || / ||z(0) - z(1)||,
             chord(t) = z(1) + (1-t) * (z(0) - z(1))
         relative deviation from the straight chord; reported as mean and max
         over t.

    path/chord = integral ||v|| dt / ||z(0) - z(1)||  (= 1/S)
         how much longer the actual path is than the straight line.

Integration is fine-grained Euler (default 200 steps) so the measurement is not
limited by discretisation; the tiny residual discretisation error is identical
for both models and both have ~200 steps, so the comparison is fair.

USAGE
    python toy_evaluation/exp4_trajectory_curvature.py --num_clips 32 --seeds 2 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_training.train_fm_cond import CondLatentDiT, get as fetch
from toy_evaluation.eval_fm_null_separation import MODELS


@torch.no_grad()
def trace(model, zmix, steps, device, seed=0):
    """Integrate t: 1 -> 0 finely, returning (Z, vnorms).

    Z      : (steps+1, B, D, T) trajectory
    vnorms : (steps, B)        ||v|| * |dt| at each step
    """
    B = zmix.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(zmix.shape, device=device, generator=g)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    pts = [z.clone()]
    vn = []
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        v = model(z, ts[i].expand(B), *((zmix,) if model.n_cond == 1 else (zmix,)))
        vn.append(v.flatten(1).norm(dim=1) * dt.abs())
        z = z + v * dt
        pts.append(z.clone())
    return torch.stack(pts), torch.stack(vn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clips", type=int, default=32)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--trace_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="toy_evaluation/exp4_fm_separation")
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    res = {}
    for label, ckpt_path, mdir in MODELS:
        mdir = Path(mdir)
        meta = json.loads((mdir / "meta.json").read_text())
        ck = torch.load(ckpt_path, map_location="cpu")
        assert ck.get("objective", "rf") == "rf", f"{label}: not an RF checkpoint"
        model = CondLatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                              depth=ck["depth"], num_heads=ck["num_heads"],
                              n_cond=int(ck.get("n_cond", 1))).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        va_mix = np.load(meta["splits"]["val"]["files"]["zmix"], mmap_mode="r")
        n = min(args.num_clips, len(va_mix))

        acc = {k: [] for k in ("S", "path_over_chord", "dev_mean", "dev_max", "chord_len")}
        with torch.no_grad():
            for sd in range(args.seeds):
                for i in range(0, n, args.batch_size):
                    zm = fetch(va_mix, slice(i, min(i + args.batch_size, n)), device)
                    Z, vn = trace(model, zm, args.trace_steps, device, seed=sd)
                    B = zm.shape[0]
                    z1, z0 = Z[0], Z[-1]
                    chord = (z0 - z1).flatten(1)
                    chord_len = chord.norm(dim=1)
                    path_len = vn.sum(0)
                    tvals = torch.linspace(1.0, 0.0, args.trace_steps + 1, device=device)
                    w = (1.0 - tvals).view(-1, 1, 1, 1)
                    chord_pts = z1.unsqueeze(0) + w * (z0 - z1).unsqueeze(0)
                    dev = ((Z - chord_pts).flatten(2).norm(dim=2) /
                           (chord_len.unsqueeze(0) + 1e-12))
                    acc["S"] += (chord_len / (path_len + 1e-12)).cpu().tolist()
                    acc["path_over_chord"] += (path_len / (chord_len + 1e-12)).cpu().tolist()
                    acc["dev_mean"] += dev.mean(0).cpu().tolist()
                    acc["dev_max"] += dev.max(0).values.cpu().tolist()
                    acc["chord_len"] += chord_len.cpu().tolist()
                print(f"  [{label}] seed {sd+1}/{args.seeds} done", flush=True)

        res[label] = {k: float(np.mean(v)) for k, v in acc.items()}
        res[label]["S_std"] = float(np.std(acc["S"]))
        res[label]["dev_max_max"] = float(np.max(acc["dev_max"]))
        res[label]["n"] = len(acc["S"])
        print(f"[{label}] S = {res[label]['S']:.4f} | path/chord "
              f"{res[label]['path_over_chord']:.4f} | dev mean "
              f"{res[label]['dev_mean']:.4f} | dev max {res[label]['dev_max']:.4f}")

    A, B = MODELS[0][0], MODELS[1][0]
    from scipy import stats
    t, pv = stats.ttest_ind_from_stats(
        res[A]["S"], res[A]["S_std"], res[A]["n"],
        res[B]["S"], res[B]["S_std"], res[B]["n"], equal_var=False)
    print(f"\n{'=' * 78}")
    print("Trajectory straightness (fine Euler, "
          f"{args.trace_steps} steps, {res[A]['n']} traces/model)")
    print(f"{'=' * 78}")
    print(f"{'quantity':<28}{A:>16}{B:>16}   (1.0 = perfectly straight)")
    print("-" * 78)
    for k, name in (("S", "straightness S"), ("path_over_chord", "path / chord"),
                    ("dev_mean", "chord deviation mean"), ("dev_max", "chord deviation max"),
                    ("chord_len", "endpoint distance ||z0-z1||")):
        print(f"{name:<28}{res[A][k]:>16.4f}{res[B][k]:>16.4f}")
    print("-" * 78)
    print(f"S: {A} {res[A]['S']:.4f} vs {B} {res[B]['S']:.4f} "
          f"(Welch t={t:.2f}, p={pv:.2e})")
    print("S closer to 1 => straighter learned field => fewer sampling steps needed")

    (out / "fm_trajectory_curvature.json").write_text(json.dumps(
        {"protocol": vars(args), "results": res,
         "welch_t": float(t), "welch_p": float(pv)}, indent=2))
    print(f"\nSaved: {out / 'fm_trajectory_curvature.json'}")


if __name__ == "__main__":
    main()
