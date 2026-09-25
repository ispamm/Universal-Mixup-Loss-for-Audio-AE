#!/usr/bin/env python3
"""
Listening demo for the AE latent algebra (no FM needed).

For a pair of real validation clips (A, B) this writes:
  * endpoint reconstructions        A_recon.wav, B_recon.wav
  * the AE latent-cone morph        ae_trig_<k>.wav     (11-step sweep)
  * the ideal audio crossfade       ideal_trig_<k>.wav  (same sweep, ground truth)
  * a single conemix at theta=45    ae_conemix45.wav
  * an algebraic separation demo    sep_<stem>.wav vs recon_<stem>.wav  (if MUSDB chunks exist)

Listen to `ae_trig_*.wav` against `ideal_trig_*.wav`: on an additive latent the
morph should track the ideal constant-power crossfade.

Usage:
    python toy_evaluation/gen_mixup_audio_demo.py \
        --checkpoint toy_training/checkpoints/transformer_univ_warm_v2_best.pt \
        --device cuda:0 --output_dir toy_evaluation/audio_demo
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper


def peak_norm(a, target=0.9):
    m = np.abs(a).max()
    return a if m < 1e-9 else (a / m * target).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", "--data_dirs", dest="data_dirs", nargs="+",
                   default=["/data/wav48_silence_trimmed",
                            "/data/fsd50k/FSD50K.dev_audio",
                            "/data/fma_small"])
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output_dir", default="toy_evaluation/audio_demo")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    n = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    stems = collect_validation_stems(args.data_dirs, sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    # pick two clips from *different* tracks for a clearer morph
    i = int(rng.integers(0, len(stems)))
    j = (i + len(stems) // 3) % len(stems)
    a = load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy()
    b = load_stem_clip(stems[j][0], n, centre=True).squeeze().numpy()
    print(f"[Data] A = {Path(stems[i][0]).name}")
    print(f"[Data] B = {Path(stems[j][0]).name}")

    ae = ToyModelWrapper(args.checkpoint, device)
    print(f"[AE]   {Path(args.checkpoint).stem} ({ae.model_type})")

    za = ae.encode(a)
    zb = ae.encode(b)
    sf.write(out / "A_original.wav", peak_norm(a), sr)
    sf.write(out / "B_original.wav", peak_norm(b), sr)
    sf.write(out / "A_recon.wav", peak_norm(ae.decode(za, n)), sr)
    sf.write(out / "B_recon.wav", peak_norm(ae.decode(zb, n)), sr)

    # ---- 11-step constant-power cone sweep: AE latent morph vs ideal audio ----
    rows = []
    for k, alpha in enumerate(np.linspace(0, 1, 11)):
        th = alpha * math.pi / 2
        c, s = math.cos(th), math.sin(th)
        # AE: mix in latent space, decode
        z_mix = c * za + s * zb
        ae_audio = ae.decode(z_mix, n)
        # ideal: mix in the audio domain
        ideal = c * a + s * b
        sf.write(out / f"ae_trig_{k:02d}_a{alpha:.1f}.wav", peak_norm(ae_audio), sr)
        sf.write(out / f"ideal_trig_{k:02d}_a{alpha:.1f}.wav", peak_norm(ideal), sr)
        rows.append({
            "k": k, "alpha": round(float(alpha), 2),
            "ae_rms": float(np.sqrt(np.mean(ae_audio ** 2))),
            "ideal_rms": float(np.sqrt(np.mean(ideal ** 2))),
        })

    # ---- single conemix at theta = 45 deg ----
    th = math.pi / 4
    z_mix = math.cos(th) * za + math.sin(th) * zb
    sf.write(out / "ae_conemix45.wav", peak_norm(ae.decode(z_mix, n)), sr)
    sf.write(out / "ideal_conemix45.wav", peak_norm(math.cos(th) * a + math.sin(th) * b), sr)

    # ---- concatenated "A -> morph -> B" for easy listening ----
    parts = []
    for k in range(11):
        alpha = k / 10
        th = alpha * math.pi / 2
        z_mix = math.cos(th) * za + math.sin(th) * zb
        parts.append(peak_norm(ae.decode(z_mix, n)) * 0.5)
        parts.append(np.zeros(int(0.15 * sr), dtype=np.float32))
    sf.write(out / "ae_sweep_concat.wav", np.concatenate(parts), sr)

    parts = []
    for k in range(11):
        alpha = k / 10
        th = alpha * math.pi / 2
        parts.append(peak_norm(math.cos(th) * a + math.sin(th) * b) * 0.5)
        parts.append(np.zeros(int(0.15 * sr), dtype=np.float32))
    sf.write(out / "ideal_sweep_concat.wav", np.concatenate(parts), sr)

    meta = {"checkpoint": args.checkpoint, "A": Path(stems[i][0]).name,
            "B": Path(stems[j][0]).name, "sample_rate": sr,
            "clip_seconds": args.clip_seconds, "sweep": rows}
    (out / "ae_mixup_demo_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote demo audio to {out}")
    print("  listen: A_original / A_recon  (reconstruction quality)")
    print("  listen: ae_sweep_concat  vs  ideal_sweep_concat  (morph quality)")
    print("  listen: ae_trig_05_a0.5  vs  ideal_trig_05_a0.5  (midpoint morph)")


if __name__ == "__main__":
    main()
