#!/usr/bin/env python3
"""
Trigonometric-interpolation energy-preservation worker (MUSDB stems)
====================================================================

For every model of the algebraic-separation study, evaluates the **full
constant-power (trig) interpolation path** between pairs of real MUSDB18-HQ
stems:

    z_mix(α) = cos(θ)·z_A + sin(θ)·z_B ,   θ = α·π/2 ,   α ∈ [0, 1]
    ideal(α) = cos(θ)·A    + sin(θ)·B      (audio domain, exact law)

and, per (pair, α), records

  * energy preservation : ratio = RMS(D(z_mix)) / RMS(ideal)
                          (1.0 = the decoded mix carries exactly the ideal
                          constant-power energy at that α),
  * lims-style normalized energies : dec_norm = RMS(D(z_mix))/RMS_mean,
                          ideal_norm = RMS(ideal)/RMS_mean,
                          RMS_mean = (RMS(A)+RMS(B))/2,
  * mix fidelity : SI-SDR and multi-scale STFT MSE of D(z_mix) vs the ideal.

Reconstruction anchors are computed on the SAME two stems (identical clips) so
all cross-model comparisons can be normalized to each model's own
reconstruction, exactly like ``aggregate.py`` does for separation:
recon_SI-SDR / recon_MS (mean of the two endpoints) and the single-clip energy
fidelity recon_rms_err = mean |RMS(recon)/RMS(orig) − 1|.

Pairs: all 6 same-track stem pairs (drums/bass/other/vocals) over the same 100
tracks (seed 42) used by ``prepare_chunks.py``  →  600 trials per model.

One script, four models — lazy imports so it runs in all three environments:

    venv      : python worker_energy_interp.py --model same
    venv      : python worker_energy_interp.py --model same-l
    cae_env   : python worker_energy_interp.py --model lin-cae
    .venv-dac : python worker_energy_interp.py --model dac

Usage (example):
    python toy_evaluation/separation_compare/worker_energy_interp.py \
        --model same --device cuda:3 --steps 11 \
        --out toy_evaluation/separation_compare/results_energy_same.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]
# all unordered same-track stem pairs
PAIRS = [
    ("drums", "bass"), ("drums", "other"), ("drums", "vocals"),
    ("bass", "other"), ("bass", "vocals"), ("other", "vocals"),
]
DECODE_SEED_BASE = 1234


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def safe_rel_err(est_rms: float, ref_rms: float) -> float:
    """|est/ref − 1| with a silence guard (NaN for silent references)."""
    return abs(est_rms / ref_rms - 1.0) if ref_rms > 1e-12 else float("nan")


def common_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(a), len(b))
    return a[:L], b[:L]


# =============================================================================
#  Per-model adapters (identical model I/O to the separation workers)
# =============================================================================

class _ToyAdapter:
    """SAME toy transformer (16 kHz), project venv."""

    name = "same"
    suffix = "16"
    sr = 16000

    def __init__(self, device: torch.device, ckpt: str):
        from toy_evaluation.model_utils import ToyModelWrapper
        self.model = ToyModelWrapper(ckpt, device)
        self.device = device

    def encode(self, x: np.ndarray) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, z: torch.Tensor, n: int, seed: int | None = None) -> np.ndarray:
        out = self.model.decode(z, n)
        return out.astype(np.float32)


class _SameLAdapter:
    """Original SAME autoencoder (44.1 kHz), project venv (stereo-in, mono-out)."""

    name = "same-l"
    suffix = "44"
    sr = 44100

    def __init__(self, device: torch.device, model_name: str = "same-l"):
        from stable_audio_3 import AutoencoderModel
        self.device = device
        self.model = AutoencoderModel.from_pretrained(model_name, device=str(device))
        self.sr = int(self.model.sample_rate)

    def encode(self, x: np.ndarray) -> torch.Tensor:
        audio_t = torch.from_numpy(np.stack([x, x]).astype(np.float32)).to(self.device)
        with torch.inference_mode():
            return self.model.encode(audio_t, sr=self.sr)  # (1, D, T_latent)

    def decode(self, z: torch.Tensor, n: int, seed: int | None = None) -> np.ndarray:
        with torch.inference_mode():
            out = self.model.decode(z)  # (1, C, samples)
        mono = out.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
        mono, _ = common_len(mono, np.zeros(n, dtype=np.float32))
        return mono


class _LinCaeAdapter:
    """Lin-CAE consistency autoencoder (44.1 kHz), cae_env (stochastic decode)."""

    name = "lin-cae"
    suffix = "44"
    sr = 44100

    def __init__(self, device: torch.device, model_id: str = "lin-cae", ckpt_type: str = "last"):
        from linear_cae import Autoencoder
        self.device = device
        self.model = Autoencoder.from_pretrained(model_id, ckpt_type=ckpt_type).to(device).eval()
        print(f"  lin-cae params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    def encode(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(torch.from_numpy(x).unsqueeze(0).to(self.device))

    def decode(self, z: torch.Tensor, n: int, seed: int | None = None) -> np.ndarray:
        torch.manual_seed(seed if seed is not None else 0)
        with torch.no_grad():
            out = self.model.decode(z, full_length=n)
        return out.squeeze(0).cpu().numpy().astype(np.float32)


class _DacAdapter:
    """DAC 44 kHz RVQ-quantized pipeline (44.1 kHz), .venv-dac."""

    name = "dac"
    suffix = "44"
    sr = 44100

    def __init__(self, device: torch.device):
        import dac
        self.device = device
        path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(path)
        self.model.to(device).eval()

    def encode(self, x: np.ndarray) -> torch.Tensor:
        audio_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            z, _codes, _latents, _cl, _cbl = self.model.encode(audio_t)
        return z  # quantized continuous rep (1, 1024, T_latent)

    def decode(self, z: torch.Tensor, n: int, seed: int | None = None) -> np.ndarray:
        with torch.inference_mode():
            out = self.model.decoder(z)
        mono = out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        mono, _ = common_len(mono, np.zeros(n, dtype=np.float32))
        return mono


class _Toy441Adapter(_ToyAdapter):
    """SAME toy transformer at 44.1 kHz (project venv) — same wrapper, 44k chunks."""

    name = "toy441"
    suffix = "44"
    sr = 44100


def load_adapter(model_name: str, device: torch.device, ckpt: str | None) -> object:
    if model_name == "same":
        return _ToyAdapter(device, ckpt)
    if model_name == "toy441":
        return _Toy441Adapter(device, ckpt)
    if model_name == "same-l":
        return _SameLAdapter(device)
    if model_name == "lin-cae":
        return _LinCaeAdapter(device)
    if model_name == "dac":
        return _DacAdapter(device)
    raise ValueError(f"unknown model: {model_name}")


# =============================================================================
#  Core
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["same", "toy441", "same-l", "lin-cae", "dac"])
    ap.add_argument("--chunks", type=Path, default=_THIS_FILE.parent / "chunks")
    ap.add_argument("--out", type=Path,
                    default=_THIS_FILE.parent / "results_energy_same.json")
    ap.add_argument("--ckpt", type=str,
                    default="toy_training/checkpoints/transformer_best_full_best.pt",
                    help="Checkpoint for --model same only.")
    ap.add_argument("--steps", type=int, default=11,
                    help="Number of α points on [0, 1] (default 11 = lims convention).")
    ap.add_argument("--max_tracks", type=int, default=None,
                    help="Limit to the first N tracks (smoke testing).")
    ap.add_argument("--device", type=str, default="cuda:3")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    sr_key = "sr16" if args.model == "same" else "sr44"
    print(f"worker {args.model} | sr={manifest[sr_key]} device={device} "
          f"tracks={len(manifest['tracks'])} steps={args.steps}")

    adapter = load_adapter(args.model, device, args.ckpt)
    print(f"  model={adapter.name} adapter={type(adapter).__name__}")

    alphas = np.linspace(0.0, 1.0, args.steps).tolist()
    trials = []
    seed_counter = 0

    tracks = manifest["tracks"]
    if args.max_tracks is not None:
        tracks = tracks[: args.max_tracks]

    for idx, track in enumerate(tracks):
        tdir = args.chunks / f"{idx:03d}"
        n = len(np.load(tdir / f"mix{adapter.suffix}.npy"))

        for (sa, sb) in PAIRS:
            a = np.load(tdir / f"{sa}{adapter.suffix}.npy")
            b = np.load(tdir / f"{sb}{adapter.suffix}.npy")
            a, b = common_len(a, b)
            n = min(n, len(a))

            z_a = adapter.encode(a)
            z_b = adapter.encode(b)
            if z_a.shape[-1] != z_b.shape[-1]:
                mlen = min(z_a.shape[-1], z_b.shape[-1])
                z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]

            # ---- reconstruction anchors (same two stems as the mixes) ----
            recon_a = adapter.decode(z_a, n, seed=DECODE_SEED_BASE + seed_counter); seed_counter += 1
            recon_b = adapter.decode(z_b, n, seed=DECODE_SEED_BASE + seed_counter); seed_counter += 1
            # decoders may return a slightly different length than n (e.g. DAC);
            # trim the reference to each decode's actual length for metric calls
            L_a, L_b = len(recon_a), len(recon_b)
            a_r, b_r = a[:L_a], b[:L_b]
            recon_sisdr = float(np.nanmean([si_sdr(a_r, recon_a), si_sdr(b_r, recon_b)]))
            recon_ms = float(np.nanmean([
                ms_stft_mse(recon_a, a_r, device),
                ms_stft_mse(recon_b, b_r, device),
            ]))
            recon_rms_err = float(np.nanmean([
                safe_rel_err(rms(recon_a), rms(a_r)),
                safe_rel_err(rms(recon_b), rms(b_r)),
            ]))

            t = {
                "track": track["id"],
                "pair": f"{sa}/{sb}",
                "alphas": alphas,
                "energy_ratio": [],      # RMS(D(trigMix z))/RMS(ideal trig mix)
                "dec_norm": [],          # RMS(D(...))/rms_mean
                "ideal_norm": [],        # RMS(ideal)/rms_mean
                "mix_sisdr": [],
                "mix_ms": [],
                "recon_sisdr": recon_sisdr,
                "recon_ms": recon_ms,
                "recon_rms_err": recon_rms_err,
            }

            for alpha in alphas:
                theta = alpha * (np.pi / 2.0)
                c, s = float(np.cos(theta)), float(np.sin(theta))
                z_mix = c * z_a + s * z_b
                dec = adapter.decode(z_mix, n, seed=DECODE_SEED_BASE + seed_counter); seed_counter += 1
                L = len(dec)
                a_L, b_L = a[:L], b[:L]
                rms_mean_L = (rms(a_L) + rms(b_L)) / 2.0
                ideal = (c * a_L + s * b_L).astype(np.float32)

                r_dec, r_ideal = rms(dec), rms(ideal)
                t["energy_ratio"].append(r_dec / r_ideal if r_ideal > 1e-12 else float("nan"))
                t["dec_norm"].append(r_dec / rms_mean_L if rms_mean_L > 1e-12 else float("nan"))
                t["ideal_norm"].append(r_ideal / rms_mean_L if rms_mean_L > 1e-12 else float("nan"))
                t["mix_sisdr"].append(si_sdr(ideal, dec))
                t["mix_ms"].append(ms_stft_mse(dec, ideal, device))

            trials.append(t)
        if (idx + 1) % 10 == 0 or idx == len(manifest["tracks"]) - 1:
            print(f"  [{idx + 1}/{len(manifest['tracks'])}] tracks done, {len(trials)} trials")

    out = {
        "model": args.model,
        "ckpt": args.ckpt if args.model == "same" else None,
        "sr": manifest[sr_key],
        "interp": "trig constant-power: z_mix = cos(α·π/2)·z_A + sin(α·π/2)·z_B",
        "alphas": alphas,
        "num_pairs_per_track": len(PAIRS),
        "num_trials": len(trials),
        "timestamp": datetime.now().isoformat(),
        "trials": trials,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{len(trials)} trials -> {args.out}")


if __name__ == "__main__":
    main()
