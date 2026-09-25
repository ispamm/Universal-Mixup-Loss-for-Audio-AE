#!/usr/bin/env python3
"""
Decoder commutation MSE — the decoder-side analogue of encoder homomorphism.

Encoder homomorphism (already in the battery):
    enc_{trig,lerp}_mse = MSE( E(mix(A,B)),  mix(E(A), E(B)) )

Decoder commutation (this script):
    dec_{trig,lerp}_mse = MSE( D(mix(Z_A,Z_B)),  mix(D(Z_A), D(Z_B)) )

i.e. "does the decoder commute with mixing?" — if the decoder were exactly
additive, the MSE would be 0. This is the waveform-space decoder additivity
metric that parallels the encoder MSE, complementing the LSD-based
`dec_*_lsd` metrics already in the battery (LSD measures end-to-end mix
fidelity vs the ideal audio mix; this MSE measures decoder commutation).

Usage:
    python toy_evaluation/eval_decoder_mse.py \
        --checkpoints ckpt1 ckpt2 ... --num_pairs 500 --device cuda:0
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip, generate_pairs
from toy_evaluation.model_utils import ToyModelWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", nargs="+",
                   default=["/data/wav48_silence_trimmed",
                            "/data/fsd50k/FSD50K.dev_audio",
                            "/data/fma_small"])
    p.add_argument("--num_pairs", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--clip_seconds", type=float, default=2.0)
    p.add_argument("--output_dir", default="toy_evaluation/eval_decoder_mse")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    clip_samples = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] {args.num_pairs} pairs, seed {args.seed}, sr {sr}, "
          f"{args.clip_seconds}s clips", flush=True)
    stems = collect_validation_stems(args.data_dirs, sample_rate=sr)
    pairs = generate_pairs(stems, args.num_pairs, seed=args.seed, mode="random")
    print(f"[Setup] {len(stems)} stems, {len(pairs)} pairs", flush=True)

    wrappers = {}
    for ck in args.checkpoints:
        wrappers[Path(ck).stem] = ToyModelWrapper(ck, device)
    print(f"[Setup] Loaded {len(wrappers)} models", flush=True)

    results = {name: {"trig": [], "lerp": []} for name in wrappers}

    for pi, (ia, ib) in enumerate(pairs):
        pa = stems[ia][0]
        pb = stems[ib][0]
        audio_a = load_stem_clip(pa, clip_samples, centre=True).squeeze().numpy()
        audio_b = load_stem_clip(pb, clip_samples, centre=True).squeeze().numpy()
        xa = torch.from_numpy(audio_a).unsqueeze(0).unsqueeze(0).to(device)
        xb = torch.from_numpy(audio_b).unsqueeze(0).unsqueeze(0).to(device)

        for name, w in wrappers.items():
            za = w.encode(audio_a)   # (1, D, T_lat)
            zb = w.encode(audio_b)
            # decode endpoints
            da = torch.from_numpy(w.decode(za, clip_samples)).unsqueeze(0).unsqueeze(0).to(device)
            db = torch.from_numpy(w.decode(zb, clip_samples)).unsqueeze(0).unsqueeze(0).to(device)

            # ---- trig at alpha=0.5: cos(pi/4), sin(pi/4) ----
            ct, st = math.cos(math.pi / 4), math.sin(math.pi / 4)
            z_trig = ct * za + st * zb
            d_trig = torch.from_numpy(w.decode(z_trig, clip_samples)).unsqueeze(0).unsqueeze(0).to(device)
            mix_d_trig = ct * da + st * db
            results[name]["trig"].append(
                torch.mean((d_trig - mix_d_trig) ** 2).item())

            # ---- lerp at alpha=0.5 ----
            a = 0.5
            z_lerp = (1 - a) * za + a * zb
            d_lerp = torch.from_numpy(w.decode(z_lerp, clip_samples)).unsqueeze(0).unsqueeze(0).to(device)
            mix_d_lerp = (1 - a) * da + a * db
            results[name]["lerp"].append(
                torch.mean((d_lerp - mix_d_lerp) ** 2).item())

        if (pi + 1) % 100 == 0:
            print(f"  [{pi+1}/{len(pairs)}] pairs done", flush=True)

    # ---- Summary ----
    print(f"\n{'='*90}", flush=True)
    print(f"Decoder commutation MSE (n={len(pairs)} pairs, α=0.5, seed {args.seed})", flush=True)
    print(f"  MSE( D(mix(Z_A,Z_B)), mix(D(Z_A),D(Z_B)) ) — lower = more additive", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Checkpoint':<38} {'dec_trig_mse':<22} {'dec_lerp_mse':<22}", flush=True)
    print("-"*90, flush=True)
    summary = {}
    for name in wrappers:
        t = np.array(results[name]["trig"]); l = np.array(results[name]["lerp"])
        summary[name] = {
            "dec_trig_mse@0.5": float(t.mean()), "dec_trig_mse@0.5_std": float(t.std()),
            "dec_lerp_mse@0.5": float(l.mean()), "dec_lerp_mse@0.5_std": float(l.std()),
            "n_pairs": len(pairs),
        }
        print(f"{name:<38} {t.mean():.6f} ± {t.std():.6f}    {l.mean():.6f} ± {l.std():.6f}", flush=True)

    out_file = out / "decoder_mse_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
