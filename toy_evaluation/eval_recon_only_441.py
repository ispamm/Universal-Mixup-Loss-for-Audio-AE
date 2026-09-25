#!/usr/bin/env python3
"""
44.1 kHz toy-AE reconstruction-only comparison (fast)
=====================================================

Compares single-audio reconstruction quality of two 44.1 kHz toy checkpoints:

  * ``univ_dt2048_441_best.pt``     — patch_size=2048 (aggressive downsampling)
  * ``transformer_univ_441_best.pt`` — patch_size=256  (normal resolution)

Metrics (per model, 1000 clips, recon vs original):
  - MS-STFT : training-matched ``SAMELoss`` magnitude loss (spectral contrast +
              adaptive log-magnitude, summed over 7 FFT resolutions 32→2048). Lower=better.
  - SI-SDR  : scale-invariant SDR in dB (Le Roux et al. 2019). Higher=better.

Also saves a handful of original/reconstructed .wav pairs per model for listening.

Usage:
    python toy_evaluation/eval_recon_only_441.py \
        --checkpoints toy_training/checkpoints/univ_dt2048_441_best.pt \
                      toy_training/checkpoints/transformer_univ_441_best.pt \
        --labels dt2048 normal \
        --num_clips 1000 \
        --audio_out toy_evaluation/audio_441 \
        --out toy_evaluation/results_441/recon_comparison.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import CLIP_DURATION, collect_validation_stems, stem_to_numpy
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss

SR44 = 44100
CLIP_SAMPLES = int(CLIP_DURATION * SR44)  # 88200


def ms_stft_loss(pred_np, tgt_np, loss_fn, device):
    p = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    t = torch.from_numpy(tgt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, mag = loss_fn(p, t, use_phase=False, return_components=True)
    return float(mag.item())


def si_sdr(reference, estimate):
    ref = reference.astype(np.float64); est = estimate.astype(np.float64)
    ref = ref - ref.mean(); est = est - est.mean()
    d = np.dot(ref, ref)
    if d < 1e-12:
        return float("nan")
    a = np.dot(est, ref) / d
    et = a * ref; en = est - et
    return float(10.0 * np.log10(np.dot(et, et) / (np.dot(en, en) + 1e-12)))


def _align(r, n):
    if len(r) > n: return r[:n]
    if len(r) < n: return np.pad(r, (0, n - len(r)))
    return r


def save_wav(path, audio, sr=SR44):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def evaluate(ckpt, device, stems, clip_indices, loss_fn, audio_out, n_save, label):
    t0 = time.time()
    model = ToyModelWrapper(ckpt, device)
    name = label or Path(ckpt).stem
    print(f"\n=== {name} ({model.model_type}) ===")
    msstft, sisdr = [], []
    saved = 0
    for idx in tqdm(clip_indices, desc=f"recon[{name}]", unit="clip"):
        path, stem, track = stems[idx]
        try:
            audio = stem_to_numpy(path, CLIP_SAMPLES, centre=True)
        except Exception as exc:
            print(f"  skip {path}: {exc}"); continue
        n = len(audio)
        r = _align(model.decode(model.encode(audio), n), n)
        msstft.append(ms_stft_loss(r, audio, loss_fn, device))
        sisdr.append(si_sdr(audio, r))
        if audio_out is not None and saved < n_save:
            dom = track.split("::")[0]
            base = f"recon_{name}_{saved:03d}_{dom}_{Path(path).stem}"
            save_wav(audio_out / name / f"{base}_orig.wav", audio)
            save_wav(audio_out / name / f"{base}_recon.wav", r)
            saved += 1
    res = {
        "label": name, "checkpoint": ckpt, "model_type": model.model_type,
        "n": len(msstft),
        "msstft_mean": float(np.mean(msstft)) if msstft else float("nan"),
        "msstft_std": float(np.std(msstft)) if msstft else float("nan"),
        "msstft_median": float(np.median(msstft)) if msstft else float("nan"),
        "sisdr_mean": float(np.nanmean(sisdr)) if sisdr else float("nan"),
        "sisdr_std": float(np.nanstd(sisdr)) if sisdr else float("nan"),
        "sisdr_median": float(np.nanmedian(sisdr)) if sisdr else float("nan"),
        "per_clip_msstft": msstft, "per_clip_sisdr": sisdr,
        "elapsed_s": time.time() - t0,
    }
    print(f"  MS-STFT={res['msstft_mean']:.6f} ± {res['msstft_std']:.6f} (median {res['msstft_median']:.6f})")
    print(f"  SI-SDR={res['sisdr_mean']:.2f} ± {res['sisdr_std']:.2f} dB (median {res['sisdr_median']:.2f})  n={res['n']}  {res['elapsed_s']:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--num_clips", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--audio_out", type=Path, default=None)
    ap.add_argument("--n_save", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    stems = collect_validation_stems(sample_rate=SR44)
    print(f"validation stems @ {SR44} Hz: {len(stems)}")
    rng = random.Random(args.seed)
    n = min(args.num_clips, len(stems))
    clip_indices = rng.sample(range(len(stems)), n)
    print(f"clips={n} seed={args.seed}")
    loss_fn = SAMELoss().to(device)

    labels = args.labels or [Path(c).stem for c in args.checkpoints]
    results = []
    for ck, lb in zip(args.checkpoints, labels):
        results.append(evaluate(ck, device, stems, clip_indices, loss_fn, args.audio_out, args.n_save, lb))

    # comparison table
    print("\n" + "=" * 90)
    print("44.1 kHz reconstruction-only comparison (1000 clips)")
    print("=" * 90)
    hdr = f"{'Metric':<28}" + "".join(f"{r['label'][:20]:>22}" for r in results)
    print(hdr); print("-" * 90)
    def row(lbl, key, fmt="{:>22.6f}"):
        print(f"{lbl:<28}" + "".join(fmt.format(r[key]) for r in results))
    row("MS-STFT (mean)", "msstft_mean")
    row("MS-STFT (std)", "msstft_std")
    row("MS-STFT (median)", "msstft_median")
    row("SI-SDR mean (dB)", "sisdr_mean", "{:>22.2f}")
    row("SI-SDR std (dB)", "sisdr_std", "{:>22.2f}")
    row("SI-SDR median (dB)", "sisdr_median", "{:>22.2f}")
    print("-" * 90)
    print(f"{'n clips':<28}" + "".join(f"{r['n']:>22}" for r in results))
    print("=" * 90)

    if len(results) == 2:
        a, b = results[0], results[1]
        print(f"\nRelative: [{b['label']}] vs [{a['label']}]")
        print(f"  MS-STFT Δ = {b['msstft_mean'] - a['msstft_mean']:+.6f}  (negative = {b['label']} better)")
        print(f"  SI-SDR Δ  = {b['sisdr_mean'] - a['sisdr_mean']:+.2f} dB  (positive = {b['label']} better)")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "44.1 kHz reconstruction-only comparison (MS-STFT + SI-SDR)",
        "timestamp": datetime.now().isoformat(), "sample_rate": SR44,
        "clip_duration_s": CLIP_DURATION, "num_clips": n, "seed": args.seed,
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"📄 JSON -> {args.out}")


if __name__ == "__main__":
    main()
