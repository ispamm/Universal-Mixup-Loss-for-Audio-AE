#!/usr/bin/env python3
"""
Prepare MUSDB18-HQ chunks for the SAME-vs-LinCAE algebraic separation study.
================================================================================

Selects *num_tracks* tracks from MUSDB18-HQ (train + test) deterministically
(seed 42), extracts a mono 2-second centre chunk of the full mixture and of the
4 stems (drums / bass / other / vocals), and stores:

  * 44.1 kHz float32 numpy arrays (native MUSDB rate)  -> for lin-cae,
  * 16 kHz   float32 numpy arrays (SAME model rate) -> for the SAME model,

plus a ``manifest.json`` consumed by both model workers.

The mixture in MUSDB18-HQ is the sum of the 4 stems (verified: max abs error
~1e-4), so the *waveform-domain* separation target ``stem = mix - sum(others)``
is exact: any separation error measured by the workers comes from the latent
algebra (encode/decode linearity), not from the dataset.

Usage (venv):
    python evaluation/separation_compare/prepare_chunks.py \
        --musdb /data/musdb18_hq \
        --out   evaluation/separation_compare/chunks \
        --num_tracks 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

STEMS = ["drums", "bass", "other", "vocals"]
SR44 = 44100
SR16 = 16000
CLIP_SECONDS = 2.0


def read_mono_clip(wav_path: Path, clip_samples: int) -> np.ndarray:
    """Read a wav, mix to mono (mean of channels), extract the centre window."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (T, ch)
    if sr != SR44:
        raise ValueError(f"Unexpected sample rate {sr} in {wav_path}")
    mono = data.mean(axis=1).astype(np.float32)
    n = len(mono)
    if n <= clip_samples:
        pad = clip_samples - n
        left = pad // 2
        mono = np.pad(mono, (left, pad - left))
        start = 0
    else:
        start = (n - clip_samples) // 2
        mono = mono[start : start + clip_samples]
    return mono, start


def resample16(x44: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x44.astype(np.float32))
    t16 = torchaudio.functional.resample(t, SR44, SR16)
    return t16.numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb", type=Path, default=Path("/data/musdb18_hq"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "chunks")
    ap.add_argument("--num_tracks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- collect all track dirs (train + test) ----
    track_dirs: list[Path] = []
    for split in ("train", "test"):
        split_dir = args.musdb / split
        if not split_dir.is_dir():
            print(f"[warn] missing split dir: {split_dir}")
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and (d / "mixture.wav").exists():
                track_dirs.append(d)
    print(f"found {len(track_dirs)} tracks under {args.musdb}")

    rng = random.Random(args.seed)
    chosen = rng.sample(track_dirs, min(args.num_tracks, len(track_dirs)))
    print(f"selected {len(chosen)} tracks (seed={args.seed})")

    clip44 = int(SR44 * CLIP_SECONDS)
    clip16 = int(SR16 * CLIP_SECONDS)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "description": "MUSDB18-HQ chunks for SAME vs Lin-CAE algebraic separation",
        "seed": args.seed,
        "num_tracks": len(chosen),
        "clip_seconds": CLIP_SECONDS,
        "sr44": SR44,
        "sr16": SR16,
        "stems": STEMS,
        "tracks": [],
    }

    for idx, track_dir in enumerate(chosen):
        track_id = f"{idx:03d}__{track_dir.parent.name}__{track_dir.name}"
        tdir = args.out / f"{idx:03d}"
        tdir.mkdir(parents=True, exist_ok=True)

        mix44, start = read_mono_clip(track_dir / "mixture.wav", clip44)
        np.save(tdir / "mix44.npy", mix44)
        np.save(tdir / "mix16.npy", resample16(mix44))
        stems = {}
        for stem in STEMS:
            s44, _ = read_mono_clip(track_dir / f"{stem}.wav", clip44)
            stems[stem] = s44
            np.save(tdir / f"{stem}44.npy", s44)
            np.save(tdir / f"{stem}16.npy", resample16(s44))

        manifest["tracks"].append({
            "id": track_id,
            "dir": str(track_dir),
            "offset44": start,
        })
        if (idx + 1) % 20 == 0 or idx == len(chosen) - 1:
            print(f"  [{idx+1}/{len(chosen)}] {track_id}")

    manifest_file = args.out / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"\nchunks -> {args.out}")
    print(f"manifest -> {manifest_file}")


if __name__ == "__main__":
    main()
