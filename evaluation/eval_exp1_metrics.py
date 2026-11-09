#!/usr/bin/env python3
"""
Exp.1 encoder/decoder additivity metrics -- per-pair statistics.

Fixes two issues with the first version:
  * it averaged *batch* means instead of per-pair values, so no valid std /
    significance test was possible;
  * it used 300 pairs while the latent-space columns came from the battery's
    1000-pair set, mixing sample sizes inside one table.

This version accumulates one value PER PAIR, so it reports mean +- std and the
standard error of the mean (SEM), saves the raw per-pair arrays (so any test can
be recomputed), runs paired significance tests against a reference model, and
can print the SEM-vs-n convergence curve used to justify the sample size.

Metrics (see eval_iclr_exp1/exp1_table.tex for the formulas):
  encoder: single (gain grid), lerp@alpha=0.5, lerp smoothness (Var over alpha)
  decoder: identical three

Usage:
    python evaluation/eval_exp1_metrics.py \
        --checkpoints ckpt1 ckpt2 ... \
        --num_pairs 1000 --device cuda:0 \
        --ref transformer_univ_warm_v2_best
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data import collect_validation_stems, load_stem_clip, generate_pairs
from evaluation.model_utils import AutoencoderWrapper


GAINS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
ALPHAS = np.linspace(0.0, 1.0, 11)
SM_ALPHAS = np.linspace(0.0, 1.0, 11)   # smoothing-curve grid (same 11 steps)


class BatchedAE:
    def __init__(self, ckpt, device):
        self.w = AutoencoderWrapper(ckpt, device)
        self.device = device
        self.is_transformer = self.w.model_type == "transformer"

    @torch.no_grad()
    def encode(self, x):                       # x: (B, T)
        z = self.w.model.encode(x.unsqueeze(1))
        return z.permute(0, 2, 1) if self.is_transformer else z

    @torch.no_grad()
    def decode(self, z, orig_len):
        zi = z.permute(0, 2, 1) if self.is_transformer else z
        return self.w.model.decode(zi, original_length=orig_len).squeeze(1)


def ps_mse(a, b):
    """Per-sample MSE over all non-batch dims -> (B,)."""
    return (a - b).pow(2).flatten(1).mean(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", nargs="+",
                   default=["/data/wav48_silence_trimmed",
                            "/data/fsd50k/FSD50K.dev_audio",
                            "/data/fma_small"])
    p.add_argument("--num_pairs", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--clip_seconds", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ref", type=str, default=None,
                   help="Checkpoint stem used as the paired-test reference")
    p.add_argument("--n_curve", action="store_true",
                   help="Also report SEM vs n to justify the sample size")
    p.add_argument("--output_dir", default="evaluation/eval_exp1_metrics")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    n = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # SAME pair construction as evaluate_all.py -> identical pairs to the battery
    stems = collect_validation_stems(args.data_dirs, sample_rate=sr)
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    print(f"[Setup] {len(pairs)} pairs (seed {args.seed}, mode=random) -- identical to "
          f"the eval battery's pair set", flush=True)

    A = torch.stack([torch.from_numpy(
        load_stem_clip(stems[ia][0], n, centre=True).squeeze().numpy()) for ia, _ in pairs])
    Bt = torch.stack([torch.from_numpy(
        load_stem_clip(stems[ib][0], n, centre=True).squeeze().numpy()) for _, ib in pairs])
    print(f"[Setup] audio {tuple(A.shape)}", flush=True)

    per_pair = {}
    for ci, ck in enumerate(args.checkpoints):
        name = Path(ck).stem
        print(f"[{ci+1}/{len(args.checkpoints)}] {name}", flush=True)
        ae = BatchedAE(ck, device)

        es, dl, ds, dd = [], [], [], []          # per-pair scalars
        ne = []                                    # per-pair norm-preservation error
        ec_curve, dc_curve = [], []               # per-pair (11,) curves

        for s in range(0, len(pairs), args.batch_size):
            xa = A[s:s+args.batch_size].to(device)
            xb = Bt[s:s+args.batch_size].to(device)

            za = ae.encode(xa)
            zb = ae.encode(xb)

            # ---- latent: constant-power norm error at alpha = 0.5 ----
            c = np.cos(np.pi / 4)
            z_mix = c * za + c * zb
            na = za.flatten(1).norm(dim=1)
            nm = z_mix.flatten(1).norm(dim=1)
            ne.append((nm - na).abs())

            # ---- encoder: gain (single) ----
            g_vals = []
            for g in GAINS:
                g_vals.append(ps_mse(ae.encode(g * xa), g * za))
            es.append(torch.stack(g_vals, 1).mean(1))     # (B,)

            # ---- encoder: lerp path ----
            ec = []
            for al in ALPHAS:
                zm = ae.encode((1 - al) * xa + al * xb)
                ec.append(ps_mse(zm, (1 - al) * za + al * zb))
            ec = torch.stack(ec, 1)                        # (B, 11)
            ec_curve.append(ec.cpu())
            dl.append(ec[:, 5])

            # ---- decoder ----
            dza = ae.decode(za, n)
            dzb = ae.decode(zb, n)
            g_vals = []
            for g in GAINS:
                g_vals.append(ps_mse(ae.decode(g * za, n), g * dza))
            ds.append(torch.stack(g_vals, 1).mean(1))

            dc = []
            for al in ALPHAS:
                dm = ae.decode((1 - al) * za + al * zb, n)
                dc.append(ps_mse(dm, (1 - al) * dza + al * dzb))
            dc = torch.stack(dc, 1)
            dc_curve.append(dc.cpu())
            dd.append(dc[:, 5])

        ec_curve = torch.cat(ec_curve, 0).numpy()          # (P, 11)
        dc_curve = torch.cat(dc_curve, 0).numpy()

        def pack(v):
            v = np.asarray(v, dtype=float)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))}

        raw = {
            "enc_single": torch.cat(es).cpu().numpy(),
            "enc_lerp@0.5": torch.cat(dl).cpu().numpy(),
            "enc_lerp_smoothness": ec_curve.var(1),
            "dec_single": torch.cat(ds).cpu().numpy(),
            "dec_lerp@0.5": torch.cat(dd).cpu().numpy(),
            "dec_lerp_smoothness": dc_curve.var(1),
            "norm_error": torch.cat(ne).cpu().numpy(),
        }
        per_pair[name] = {k: pack(v) for k, v in raw.items()}
        per_pair[name].update({
            "enc_lerp_mean_curve": ec_curve.mean(0).tolist(),
            "dec_lerp_mean_curve": dc_curve.mean(0).tolist(),
            "curve_var_of_mean_enc": float(ec_curve.mean(0).var()),
            "curve_var_of_mean_dec": float(dc_curve.mean(0).var()),
        })
        # raw per-pair arrays for downstream significance testing
        per_pair[name].update({f"raw_{k}": v.tolist() for k, v in raw.items()})
        r = per_pair[name]
        print(f"   enc single {r['enc_single']['mean']:.4f}+-{r['enc_single']['std']:.4f} "
              f"| enc lerp {r['enc_lerp@0.5']['mean']:.5f}+-{r['enc_lerp@0.5']['std']:.5f} "
              f"| dec single {r['dec_single']['mean']:.5f}+-{r['dec_single']['std']:.5f} "
              f"| dec lerp {r['dec_lerp@0.5']['mean']:.6f}+-{r['dec_lerp@0.5']['std']:.6f}",
              flush=True)
        (out / "exp1_enc_dec_metrics.json").write_text(json.dumps(per_pair, indent=2))
        del ae
        torch.cuda.empty_cache()

    # ---------------- paired significance tests ----------------
    if args.ref:
        from scipy import stats
        ref = args.ref
        if ref not in per_pair:
            print(f"[warn] reference '{ref}' not among evaluated checkpoints")
        else:
            print(f"\n{'='*100}\nPaired tests vs reference '{ref}' (n={per_pair[ref]['enc_lerp@0.5']['n']} pairs)\n{'='*100}")
            tests = {}
            for name in per_pair:
                if name == ref:
                    continue
                tests[name] = {}
                for key in ["enc_single", "enc_lerp@0.5", "enc_lerp_smoothness",
                            "dec_single", "dec_lerp@0.5", "dec_lerp_smoothness",
                            "norm_error"]:
                    a = np.asarray(per_pair[ref].get("raw_" + key, []))
                    b = np.asarray(per_pair[name].get("raw_" + key, []))
                    if a.size == 0 or b.size == 0:
                        continue
                    t, pv = stats.ttest_rel(a, b)
                    tests[name][key] = {"delta": float(b.mean() - a.mean()),
                                        "t": float(t), "p": float(pv)}
                row = tests[name]
                def fmt(k):
                    d = row.get(k)
                    if not d:
                        return "      --      "
                    return f"{d['delta']:+.2e} p={d['p']:.1e}"
                print(f"  {name:<34} encL {fmt('enc_lerp@0.5'):<24} decL {fmt('dec_lerp@0.5')}")
            (out / "exp1_paired_tests.json").write_text(json.dumps(tests, indent=2))

    # ---------------- SEM vs n ----------------
    if args.n_curve:
        print(f"\n{'='*100}\nSEM vs n (reference model), showing how the estimate stabilises\n{'='*100}")
        for name in per_pair:
            arr = np.asarray(per_pair[name]["raw_dec_lerp@0.5"])
            if arr.size == 0:
                continue
            line = [f"  {name:<34}"]
            for nn in [25, 50, 100, 200, 400, 1000]:
                if nn > arr.size:
                    continue
                sub = arr[:nn]
                line.append(f"n={nn}: {sub.std(ddof=1)/np.sqrt(nn):.2e}")
            print("  ".join(line))
        # effect size vs SEM for the key comparison
        if args.ref and args.ref in per_pair:
            print(f"\n  Effect vs noise (reference = {args.ref}, dec_lerp@0.5):")
            ra = np.asarray(per_pair[args.ref]["raw_dec_lerp@0.5"])
            from scipy import stats
            for name in per_pair:
                if name == args.ref:
                    continue
                b = np.asarray(per_pair[name]["raw_dec_lerp@0.5"])
                if b.size != ra.size:
                    continue
                diff = b - ra
                dz = diff.mean() / (diff.std(ddof=1) + 1e-30)
                nn_needed = int(np.ceil((2.8 / (abs(dz) + 1e-30)) ** 2)) if dz != 0 else -1
                print(f"    {name:<34} delta={diff.mean():+.2e}  paired d_z={dz:+.2f}  "
                      f"n for 80% power ~ {nn_needed}")

    f = out / "exp1_enc_dec_metrics.json"
    f.write_text(json.dumps(per_pair, indent=2))
    print(f"\nSaved: {f}")


if __name__ == "__main__":
    main()
