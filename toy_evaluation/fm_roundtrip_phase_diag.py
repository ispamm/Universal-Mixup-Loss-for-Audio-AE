#!/usr/bin/env python3
"""
Waveform-level diagnostics for the FM round-trip recon (supplementary to the
MS-STFT eval).

The MS-STFT eval is phase-blind (the SAMELoss magnitude terms ignore phase), so
it cannot tell us *how* the recon differs from the original.  Since the point of
the accompanying audio dump is listening, this script characterises what will
be heard.

IMPORTANT - the AEs reconstruct with inverted polarity.  A 440 Hz sine
round-trips through these AEs as ``-sine`` (r = -1.000, MS-STFT 0.067): the
magnitude spectrum is reproduced almost exactly while the waveform sign is
flipped.  That is a property of the AEs, whose training objective is the
phase-blind MS-STFT (for which the global sign is mathematically
unconstrained), and it is inaudible.  Consequently every waveform-level number
here is reported both raw (``alpha``) and on the SIGN-CORRECTED recon, and the
AE-ceiling row gives the no-FM-in-the-loop baseline for comparison.

  alpha        : least-squares gain <x, x_hat>/<x, x>.  Negative => anti-phase.
  r0           : waveform Pearson r at zero lag (sign-corrected).
  best_lag_ms  : lag maximising |cross-correlation| (+-100 ms).  Nonzero => the
                 recon is time-shifted, which *is* audible and would also cost
                 MS-STFT.
  r_best       : waveform r at that lag (sign-corrected).
  mel_r        : Pearson r between flattened log-mel of orig and recon -- does
                 the recon carry the right time-frequency envelope?
  sisdr_db     : SI-SDR (scale- and sign-invariant), a phase-sensitive reference.

USAGE
    python toy_evaluation/fm_roundtrip_phase_diag.py \
        --fm_a .../fm_univ_best.pt --fm_b .../fm_softnorm_best.pt \
        --label_a univ --label_b softnorm --num_clips 16 --steps 250
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import collect_validation_stems, load_stem_clip
from toy_evaluation.model_utils import ToyModelWrapper
from toy_evaluation.eval_fm_roundtrip_msstft import (
    load_fm, ae_encode, ae_decode, roundtrip)

KEYS = ["alpha", "r0", "lag_ms", "r_best", "mel_r", "sisdr"]


def si_sdr_db(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    a = float(np.dot(ref, est) / (np.dot(ref, ref) + 1e-8))
    tgt = a * ref
    return float(10 * np.log10((np.dot(tgt, tgt) + 1e-8) /
                               (np.dot(est - tgt, est - tgt) + 1e-8)))


def mel_r(a, b, sr=16000):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=96, power=2.0)
    A = torch.log(mel(torch.from_numpy(a).float()) + 1e-6).flatten().numpy()
    B = torch.log(mel(torch.from_numpy(b).float()) + 1e-6).flatten().numpy()
    return float(np.corrcoef(A, B)[0, 1])


def best_lag(a, b, sr, max_ms=100, stride=8):
    """(r0, best_lag_ms, r_best) maximising |correlation| over +-max_ms."""
    a = a - a.mean(); b = b - b.mean()
    n = len(a)
    r0 = float(np.corrcoef(a, b)[0, 1])
    best_r, best_l = r0, 0.0
    m = int(sr * max_ms / 1000)
    for lag in range(-m, m + 1, stride):
        if lag < 0:
            aa, bb = a[-lag:], b[:n + lag]
        elif lag > 0:
            aa, bb = a[:n - lag], b[lag:]
        else:
            continue
        den = np.linalg.norm(aa) * np.linalg.norm(bb)
        if den < 1e-12:
            continue
        r = float(np.dot(aa, bb) / den)
        if abs(r) > abs(best_r):
            best_r, best_l = r, lag / sr * 1000.0
    return r0, best_l, best_r


def waveform_diag(a, h, sr, sl):
    """Sign-corrected waveform diagnostics of recon h against original a."""
    ref = a[sl] - a[sl].mean()
    sig = h[sl] - h[sl].mean()
    alpha = float(np.dot(ref, sig) / (np.dot(ref, ref) + 1e-12))
    sc = 1.0 if alpha >= 0 else -1.0
    r0, lag, r_best = best_lag(a[sl], sc * h[sl], sr)
    return {"alpha": alpha, "r0": r0, "lag_ms": lag, "r_best": r_best,
            "mel_r": mel_r(a, sc * h, sr), "sisdr": si_sdr_db(a, sc * h)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_a", required=True)
    p.add_argument("--fm_b", required=True)
    p.add_argument("--label_a", default="univ")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--num_clips", type=int, default=16)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="toy_evaluation/eval_fm_roundtrip")
    args = p.parse_args()

    device = torch.device(args.device)
    sr, n = args.sample_rate, int(args.sample_rate * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    labels = [args.label_a, args.label_b]
    M = {}
    for lab, path in [(args.label_a, args.fm_a), (args.label_b, args.fm_b)]:
        fm, ck = load_fm(path, device)
        M[lab] = {"fm": fm, "ae": ToyModelWrapper(ck["ae_checkpoint"], device)}
        print(f"[Load] {lab:12s} FM {Path(path).name}  AE {Path(ck['ae_checkpoint']).stem}")

    stems = collect_validation_stems(sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(stems), size=args.num_clips, replace=False)

    res = {lab: {f"{k}{sfx}": [] for k in KEYS for sfx in ("", "_ceiling")}
           for lab in labels}

    for j, i in enumerate(idx):
        a = load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy()
        c = n // 2
        sl = slice(c, c + 3 * sr)
        x = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)

        for lab in labels:
            z0 = ae_encode(M[lab]["ae"], x)
            z_hat, _ = roundtrip(M[lab]["fm"], z0, args.steps)
            h = ae_decode(M[lab]["ae"], z_hat, n).squeeze().cpu().numpy()
            h_ceil = ae_decode(M[lab]["ae"], z0, n).squeeze().cpu().numpy()

            d = waveform_diag(a, h, sr, sl)
            d_c = waveform_diag(a, h_ceil, sr, sl)
            for k in KEYS:
                res[lab][k].append(d[k])
                res[lab][f"{k}_ceiling"].append(d_c[k])
        print(f"  [{j+1}/{len(idx)}]", flush=True)

    print(f"\n{'=' * 104}")
    print("Waveform-level diagnostics (supplementary; MS-STFT is the eval metric)")
    print("all waveform numbers are on the SIGN-CORRECTED recon (AEs invert polarity)")
    print(f"{'=' * 104}")
    print(f"{'model':<10}{'loop':<12}{'alpha':>9}{'r0':>8}{'lag ms':>9}{'r_best':>9}"
          f"{'mel_r':>8}{'SI-SDR dB':>11}")
    print("-" * 104)
    summary = {}
    for lab in labels:
        summary[lab] = {}
        for sfx, name in (("_ceiling", "AE ceiling"), ("", "FM roundtrip")):
            m = {k: float(np.mean(res[lab][f"{k}{sfx}"])) for k in KEYS}
            m["lag_ms_absmean"] = float(np.mean(np.abs(res[lab][f"lag_ms{sfx}"])))
            summary[lab][name] = m
            print(f"{lab:<10}{name:<12}{m['alpha']:>9.3f}{m['r0']:>8.3f}"
                  f"{m['lag_ms']:>9.2f}{m['r_best']:>9.3f}{m['mel_r']:>8.3f}"
                  f"{m['sisdr']:>11.2f}")
    print("-" * 104)
    for lab in labels:
        s = summary[lab]
        ce, rt = s["AE ceiling"], s["FM roundtrip"]
        print(f"  {lab}: AE alone {'inverts polarity' if ce['alpha'] < 0 else 'keeps polarity'}"
              f" (alpha={ce['alpha']:+.3f}); FM round trip alpha={rt['alpha']:+.3f}, "
              f"|lag| {rt['lag_ms_absmean']:.2f} ms")
        print(f"        SI-SDR {ce['sisdr']:.2f} dB (AE) -> {rt['sisdr']:.2f} dB (round trip): "
              f"FM costs {ce['sisdr'] - rt['sisdr']:.2f} dB of waveform fidelity; "
              f"mel_r {ce['mel_r']:.3f} -> {rt['mel_r']:.3f}")

    (out / "phase_diagnostics.json").write_text(json.dumps(
        {"summary": summary, "per_clip": res,
         "protocol": {"steps": args.steps, "num_clips": args.num_clips,
                      "window": "3 s centre for alpha/r0/lag; full clip for mel_r/SI-SDR",
                      "sign_corrected": True}}, indent=2))
    print(f"\nJSON: {out / 'phase_diagnostics.json'}")


if __name__ == "__main__":
    main()
