#!/usr/bin/env python3
"""
Generate audio mixtures from the two universal-mix ("univ") toy checkpoints.

For each checkpoint, for each of the 100 MUSDB18-HQ chunks prepared by
``separation_compare/prepare_chunks.py``, this script:

  * picks two stems of the track,
  * encodes both stems into latents (Z_A, Z_B),
  * forms the universal-mix cone latent

        z_mix = A * (cos(θ) * Z_A + sin(θ) * Z_B)

    with a *random* mixing angle θ ~ U(0, π/2) and a *fixed* energy scale
    A = 1 (so the 100 mixtures sweep the whole positive mixing-cone angle
    between the two stems),
  * decodes z_mix back to audio,

and writes 100 mono .wav files per checkpoint into
``toy_evaluation/audio_mixes/<ckpt_stem>/`` plus a ``manifest.json`` recording
the per-mixture track, stems, θ and A.

This directly exercises the decoder cone the universal-mix models were trained
to satisfy (``D(A·cosθ·Z_A + A·sinθ·Z_B) ≈ A·cosθ·X_A + A·sinθ·X_B``).

Usage:
    source .venv/bin/activate
    python toy_evaluation/generate_audio_mixes.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.model_utils import ToyModelWrapper  # noqa: E402

STEMS = ["drums", "bass", "other", "vocals"]

# (checkpoint path, native sample rate, chunk suffix)
# "44" -> 44.1 kHz chunks, "16" -> 16 kHz chunks
CHECKPOINTS = [
    ("toy_training/checkpoints/transformer_univ_441_best.pt", 44),
    ("toy_training/checkpoints/transformer_univ_full_best.pt", 16),
]

CHUNKS_DIR = _THIS_FILE.parent / "separation_compare" / "chunks"
OUT_DIR = _THIS_FILE.parent / "audio_mixes"


def mix_cone(z_a: torch.Tensor, z_b: torch.Tensor, theta: float, gain: float = 1.0) -> torch.Tensor:
    """Universal positive-weight cone latent: A*(cosθ·Z_A + sinθ·Z_B)."""
    c, s = math.cos(theta), math.sin(theta)
    return gain * (c * z_a + s * z_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num", type=int, default=100, help="number of mixtures per checkpoint")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for theta/stem selection")
    ap.add_argument("--gain", type=float, default=1.0, help="energy scale A (fixed)")
    args = ap.parse_args()

    device = torch.device(args.device)
    manifest_all = json.loads((args.chunks / "manifest.json").read_text())
    tracks = manifest_all["tracks"]
    n_tracks = min(args.num, len(tracks))
    print(f"Generating {n_tracks} mixtures per checkpoint "
          f"| device={device} | seed={args.seed} | gain A={args.gain}")

    # deterministic but different theta + stem-pair per track
    rng = np.random.default_rng(args.seed)
    # random mixing angle θ ~ U(0, π/2) per track
    thetas = (torch.pi / 2.0) * torch.rand(n_tracks, generator=torch.Generator().manual_seed(args.seed)).numpy()
    # two distinct stems per track (cyclic pairs to cover all 6 combos evenly)
    stem_pairs = []
    pair_list = [(STEMS[i], STEMS[j]) for i in range(len(STEMS)) for j in range(i + 1, len(STEMS))]
    for i in range(n_tracks):
        stem_pairs.append(pair_list[i % len(pair_list)])

    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    for ckpt_rel, sr_suffix in CHECKPOINTS:
        ckpt_path = str(_REPO_ROOT / ckpt_rel) if not Path(ckpt_rel).is_absolute() else ckpt_rel
        ckpt_stem = Path(ckpt_path).stem
        sr_key = f"sr{sr_suffix}"
        sr = manifest_all[sr_key]
        suffix = str(sr_suffix)
        out_sub = args.out / ckpt_stem
        out_sub.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ckpt_stem}  (sr={sr}, chunk suffix={suffix}) ===")

        model = ToyModelWrapper(ckpt_path, device)
        print(f"model: {model.name} ({model.model_type})")

        entries = []
        for idx in range(n_tracks):
            tdir = args.chunks / f"{idx:03d}"
            stem_a, stem_b = stem_pairs[idx]
            a_np = np.load(tdir / f"{stem_a}{suffix}.npy")
            b_np = np.load(tdir / f"{stem_b}{suffix}.npy")
            n = len(a_np)

            z_a = model.encode(a_np)  # (1, D, T_lat)
            z_b = model.encode(b_np)
            z_mix = mix_cone(z_a, z_b, float(thetas[idx]), gain=args.gain)
            audio = model.decode(z_mix, n)  # (T,)

            # safety clamp + normalize to avoid clipping on the positive cone
            audio = np.clip(audio, -1.0, 1.0)
            peak = float(np.max(np.abs(audio)))
            if peak > 0.99:
                audio = audio / peak * 0.99

            fname = f"mix_{idx:03d}_{stem_a}_{stem_b}.wav"
            torchaudio.save(
                str(out_sub / fname),
                torch.from_numpy(audio).float().unsqueeze(0),
                sr,
            )
            entries.append({
                "track": tracks[idx]["id"],
                "stem_a": stem_a,
                "stem_b": stem_b,
                "theta_deg": float(thetas[idx] * 180.0 / math.pi),
                "gain": float(args.gain),
                "file": fname,
                "sr": sr,
            })
            if (idx + 1) % 10 == 0 or idx == n_tracks - 1:
                print(f"  [{idx+1}/{n_tracks}] {fname}  θ={thetas[idx]*180/math.pi:.1f}°")

        manifest = {
            "checkpoint": ckpt_path,
            "checkpoint_stem": ckpt_stem,
            "sr": sr,
            "num_mixtures": n_tracks,
            "gain": float(args.gain),
            "seed": args.seed,
            "method": "universal-mix cone: A*(cosθ*Z_A + sinθ*Z_B), θ~U(0,π/2), A fixed",
            "timestamp": datetime.now().isoformat(),
            "mixtures": entries,
        }
        (out_sub / "manifest.json").write_text(json.dumps(manifest, indent=2))
        summary.append({"checkpoint_stem": ckpt_stem, "sr": sr, "num": n_tracks, "dir": str(out_sub)})

    print("\n=== Summary ===")
    for s in summary:
        print(f"  {s['checkpoint_stem']:35s} sr={s['sr']:6d}  {s['num']} mixes -> {s['dir']}")


if __name__ == "__main__":
    main()
