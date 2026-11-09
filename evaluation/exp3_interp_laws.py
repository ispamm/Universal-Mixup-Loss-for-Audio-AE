#!/usr/bin/env python3
"""
ICLR Experiment 3 — Held-out interpolation laws (44.1 kHz, generalization)
==========================================================================

Tests whether each autoencoder's decoder is **law-invariant**: given a latent
pair (z_A, z_B), decoding a positive-weight interpolation

    z_mix(α) = w1(α)·z_A + w2(α)·z_B

should reproduce the audio-domain mix  w1(α)·A + w2(α)·B  for EVERY
positive-weight law — not just the one the model trained on.

Models (44.1 kHz, trig_441 REMOVED per request):
    univ_441 — Universal Mixup (ours), trained on the full cone θ∈U(0,π/2)
    same-l   — production SAME, no additivity training
    dac      — DAC 44khz, RVQ-quantized
    lin-cae  — linear-CAE baseline, stochastic one-step decode

Interpolation laws (spanning linear → strongly non-linear; all positive-weight):
    lerp     : (1-α, α)                 — linear (in-distribution sanity)
    trig     : (cos(α·π/2), sin(α·π/2)) — constant-power (in-distribution for univ)
    logfade : (∝ -log10(1-α), ∝ -log10(α)) — equal-power audio standard (held-out)
    sqrt     : (√(1-α), √α)            — constant-energy, different from trig (held-out)
    parabolic: (1-α², α²)             — interior-concentrated (held-out)
    exp      : (exp(-kα)-norm, ...)    — endpoint-concentrated, k=3 (held-out)
    random   : (U(0,1), 1-w1)          — a different law per pair (held-out)

For every (pair, law, α) we record:
  * mix_sisdr  — SI-SDR(D(z_mix), ideal_audio_mix)   (higher = better)
  * mix_ms     — multi-scale STFT magnitude MSE      (lower = better)
  * recon_sisdr / recon_ms — each model's own reconstruction of the two stems
                             (anchor; identical clips so cross-model comparison
                              can be normalized to own reconstruction).

Reported summary (see aggregate_exp3.py):
  * path mean MS-STFT  = mean over the 9 α points of mix_ms      (per pair)
  * path mean SI-SDR   = mean over the 9 α points of mix_sisdr   (per pair)
  then mean ± std over pairs, plus std-of-mean (std/√n) to judge significance.

Runs in three environments (lazy imports, one script):
    venv      : python exp3_interp_laws.py --model univ_441 --device cuda:0
    venv      : python exp3_interp_laws.py --model same-l   --device cuda:0
    cae_env   : python exp3_interp_laws.py --model lin-cae  --device cuda:0
    .venv-dac : python exp3_interp_laws.py --model dac      --device cuda:0

Usage:
    python evaluation/exp3_interp_laws.py \\
        --model univ_441 \\
        --ckpt training/checkpoints/transformer_univ_441_best.pt \\
        --device cuda:0 --steps 9 --out evaluation/exp3/results_univ_441.json

16 kHz checkpoints (e.g. the universal-loss sampler ablation) load the
*16.npy chunks and need a distinct aggregator key each:

    python evaluation/exp3_interp_laws.py \\
        --model univ_441 --sr 16000 --label cart_iso \\
        --ckpt training/checkpoints/ablation_sampler_cart_iso_best.pt \\
        --device cuda:0 --out evaluation/exp3/results_cart_iso.json
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

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.separation_compare.separation_metrics import (  # noqa: E402
    ms_stft_mse,
    si_sdr,
)

STEMS = ["drums", "bass", "other", "vocals"]
# all unordered same-track stem pairs → 6 pairs/track × 100 tracks = 600 pairs
PAIRS = [
    ("drums", "bass"), ("drums", "other"), ("drums", "vocals"),
    ("bass", "other"), ("bass", "vocals"), ("other", "vocals"),
]
DECODE_SEED_BASE = 1234


# =============================================================================
#  Interpolation laws  (α -> (w1, w2)), all positive-weight, inside the cone
# =============================================================================

def _norm_pair(w1, w2):
    """Normalize a positive-weight pair so it sums to 1 (barycentric, keeps the
    path on the cone surface for constant-power-style laws). For lerp/trig the
    weights already sum to 1, so this is a no-op."""
    s = w1 + w2
    if s <= 1e-12:
        return 0.5, 0.5
    return w1 / s, w2 / s


# Each law: name -> (alpha -> (w1, w2)). Weights are NON-NEGATIVE for α∈[0,1].
LAW_NAMES = ["lerp", "trig", "logfade", "sqrt", "parabolic", "exp", "random"]
_LAWS = {}


def _lerp(alpha):
    return (1.0 - alpha, alpha)
_LAWS["lerp"] = _lerp


def _trig(alpha):
    t = alpha * (math.pi / 2.0)
    return (float(math.cos(t)), float(math.sin(t)))
_LAWS["trig"] = _trig


def _logfade(alpha):
    # equal-power audio fader: ∝ -log10(1-α), -log10(α). Clamp endpoints.
    eps = 1e-6
    a = min(max(alpha, eps), 1.0 - eps)
    w1 = -math.log10(1.0 - a)
    w2 = -math.log10(a)
    return _norm_pair(w1, w2)
_LAWS["logfade"] = _logfade


def _sqrt_law(alpha):
    return (math.sqrt(max(1.0 - alpha, 0.0)), math.sqrt(max(alpha, 0.0)))
_LAWS["sqrt"] = _sqrt_law


def _parabolic(alpha):
    return (1.0 - alpha * alpha, alpha * alpha)
_LAWS["parabolic"] = _parabolic


def _exp_law(alpha, k=3.0):
    # endpoint-concentrated; exp(-kα) falls from 1→e^-k, exp(-k(1-α)) rises.
    w1 = math.exp(-k * alpha)
    w2 = math.exp(-k * (1.0 - alpha))
    return _norm_pair(w1, w2)
_LAWS["exp"] = _exp_law


def _random_law(alpha, rng):
    # a different law per pair: pick w1(α) = w1_mid * shape(α), where w1_mid ~ U(0,1)
    # and shape is a random power p ~ U(0.3, 3). Kept stable per pair via the
    # pair-specific rng passed in. w2 = 1 - w1.
    w1_mid, p = rng  # (float, float) pre-sampled per pair
    w1 = w1_mid * (1.0 - alpha) ** p + (1.0 - w1_mid) * alpha ** p
    # ensure non-negative and in [0,1]
    w1 = min(max(w1, 0.0), 1.0)
    return (w1, 1.0 - w1)


def law_weights(law: str, alpha: float, rng_state=None):
    """Return (w1, w2) for a given law and α. For 'random', rng_state is the
    per-pair (w1_mid, p)."""
    if law == "random":
        return _random_law(alpha, rng_state)
    return _LAWS[law](alpha)


def sample_random_pair_params(seed: int):
    """Pre-sample the random law's per-pair parameters so the SAME random law
    is applied identically across all models (fairness). Returns (w1_mid, p)."""
    rng = np.random.RandomState(seed)
    w1_mid = float(rng.uniform(0.1, 0.9))
    p = float(rng.uniform(0.3, 3.0))
    return (w1_mid, p)


# =============================================================================
#  Per-model adapters (identical I/O to worker_energy_interp.py)
# =============================================================================

class _Ae441Adapter:
    """Transformer autoencoder (project venv).

    Chunk-rate agnostic: ``suffix``/``sr`` select which prepared chunks are
    loaded — "44"/44100 for the 44.1 kHz models (Exp. 2), "16"/16000 for the
    16 kHz models (Exp. 1/4, e.g. the sampler-ablation checkpoints).
    """

    name = "univ_441"
    suffix = "44"
    sr = 44100

    def __init__(self, device: torch.device, ckpt: str,
                 suffix: str = "44", sr: int = 44100):
        from evaluation.model_utils import AutoencoderWrapper
        self.model = AutoencoderWrapper(ckpt, device)
        self.device = device
        self.suffix = suffix
        self.sr = sr

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
        L = min(len(mono), n)
        return mono[:L]


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
        mono = out.squeeze(0).cpu().numpy().astype(np.float32)
        L = min(len(mono), n)
        return mono[:L]


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
        L = min(len(mono), n)
        return mono[:L]


def load_adapter(model_name: str, device: torch.device, ckpt: str | None,
                 suffix: str = "44", sr: int = 44100) -> object:
    if model_name == "univ_441":
        return _Ae441Adapter(device, ckpt, suffix=suffix, sr=sr)
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

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def safe_rel_err(est_rms: float, ref_rms: float) -> float:
    return abs(est_rms / ref_rms - 1.0) if ref_rms > 1e-12 else float("nan")


def common_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(a), len(b))
    return a[:L], b[:L]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["univ_441", "same-l", "lin-cae", "dac"])
    ap.add_argument("--chunks", type=Path,
                    default=_REPO_ROOT / "evaluation/separation_compare/chunks")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint path (required for --model univ_441).")
    ap.add_argument("--laws", nargs="+", default=LAW_NAMES,
                    help="Subset of laws to run (default: all 7).")
    ap.add_argument("--steps", type=int, default=9,
                    help="Number of α points on [0, 1] (default 9).")
    ap.add_argument("--max_tracks", type=int, default=None,
                    help="Limit to the first N tracks (smoke testing).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the per-pair random-law params (shared across models).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--sr", type=int, default=44100, choices=[44100, 16000],
                    help="Chunk sample rate: 44100 loads the *44.npy chunks "
                         "(default), 16000 the *16.npy chunks (16 kHz "
                         "models, e.g. the sampler-ablation checkpoints).")
    ap.add_argument("--label", type=str, default=None,
                    help="Override the model name recorded in the output JSON. "
                         "Several checkpoints share the univ_441 adapter; "
                         "distinct labels keep aggregate_exp3.py from merging "
                         "them under the same key.")
    args = ap.parse_args()

    if args.model == "univ_441" and args.ckpt is None:
        ap.error("--ckpt is required for --model univ_441")
    if args.sr != 44100 and args.model != "univ_441":
        ap.error(f"--sr {args.sr} is only supported for --model univ_441 "
                 f"(checkpoints); {args.model} is a 44.1 kHz model")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = "44" if args.sr == 44100 else "16"
    sr_key = "sr44" if args.sr == 44100 else "sr16"
    manifest = json.loads((args.chunks / "manifest.json").read_text())
    print(f"worker {args.label or args.model} | sr={manifest[sr_key]} device={device} "
          f"tracks={len(manifest['tracks'])} steps={args.steps} laws={args.laws}")

    adapter = load_adapter(args.model, device, args.ckpt, suffix=suffix, sr=args.sr)
    print(f"  model={adapter.name} adapter={type(adapter).__name__}")

    alphas = np.linspace(0.0, 1.0, args.steps).tolist()
    laws = args.laws
    print(f"  alphas ({len(alphas)}): {[round(a,3) for a in alphas]}")

    tracks = manifest["tracks"]
    if args.max_tracks is not None:
        tracks = tracks[: args.max_tracks]

    trials = []
    seed_counter = 0
    # Pre-sample the random-law params per (track, pair) deterministically from
    # the shared seed so every model sees the SAME random law (fairness).
    random_params = {}
    rp_rng = np.random.RandomState(args.seed)
    for idx, track in enumerate(tracks):
        for (sa, sb) in PAIRS:
            key = f"{idx:03d}/{sa}/{sb}"
            random_params[key] = (
                float(rp_rng.uniform(0.1, 0.9)),
                float(rp_rng.uniform(0.3, 3.0)),
            )

    for idx, track in enumerate(tracks):
        tdir = args.chunks / f"{idx:03d}"
        n0 = len(np.load(tdir / f"mix{adapter.suffix}.npy"))

        for (sa, sb) in PAIRS:
            a = np.load(tdir / f"{sa}{adapter.suffix}.npy")
            b = np.load(tdir / f"{sb}{adapter.suffix}.npy")
            a, b = common_len(a, b)
            n = min(n0, len(a))

            z_a = adapter.encode(a)
            z_b = adapter.encode(b)
            if z_a.shape[-1] != z_b.shape[-1]:
                mlen = min(z_a.shape[-1], z_b.shape[-1])
                z_a, z_b = z_a[..., :mlen], z_b[..., :mlen]

            # ---- reconstruction anchors (same two stems as the mixes) ----
            recon_a = adapter.decode(z_a, n, seed=DECODE_SEED_BASE + seed_counter); seed_counter += 1
            recon_b = adapter.decode(z_b, n, seed=DECODE_SEED_BASE + seed_counter); seed_counter += 1
            La, Lb = len(recon_a), len(recon_b)
            a_r, b_r = a[:La], b[:Lb]
            with np.errstate(invalid="ignore"):
                recon_sisdr = float(np.nanmean([si_sdr(a_r, recon_a), si_sdr(b_r, recon_b)]))
                recon_ms = float(np.nanmean([
                    ms_stft_mse(recon_a, a_r, device),
                    ms_stft_mse(recon_b, b_r, device),
                ]))
                recon_rms_err = float(np.nanmean([
                    safe_rel_err(rms(recon_a), rms(a_r)),
                    safe_rel_err(rms(recon_b), rms(b_r)),
                ]))

            rp = random_params[f"{idx:03d}/{sa}/{sb}"]

            for law in laws:
                t = {
                    "track": track["id"],
                    "pair": f"{sa}/{sb}",
                    "law": law,
                    "alphas": alphas,
                    "mix_sisdr": [],
                    "mix_ms": [],
                    "dec_norm": [],
                    "ideal_norm": [],
                    "recon_sisdr": recon_sisdr,
                    "recon_ms": recon_ms,
                    "recon_rms_err": recon_rms_err,
                }
                if law == "random":
                    t["random_params"] = {"w1_mid": rp[0], "p": rp[1]}

                for alpha in alphas:
                    w1, w2 = law_weights(law, alpha, rng_state=rp)
                    z_mix = w1 * z_a + w2 * z_b
                    dec = adapter.decode(z_mix, n, seed=DECODE_SEED_BASE + seed_counter); seed_counter += 1
                    L = len(dec)
                    a_L, b_L = a[:L], b[:L]
                    rms_mean_L = (rms(a_L) + rms(b_L)) / 2.0
                    ideal = (w1 * a_L + w2 * b_L).astype(np.float32)

                    r_dec, r_ideal = rms(dec), rms(ideal)
                    t["dec_norm"].append(r_dec / rms_mean_L if rms_mean_L > 1e-12 else float("nan"))
                    t["ideal_norm"].append(r_ideal / rms_mean_L if rms_mean_L > 1e-12 else float("nan"))
                    t["mix_sisdr"].append(si_sdr(ideal, dec))
                    t["mix_ms"].append(ms_stft_mse(dec, ideal, device))

                trials.append(t)

        if (idx + 1) % 10 == 0 or idx == len(tracks) - 1:
            print(f"  [{idx + 1}/{len(tracks)}] tracks done, {len(trials)} trials "
                  f"({len(trials)/(idx+1):.1f}/track)")

    out = {
        "model": args.label or args.model,
        "ckpt": args.ckpt,
        "sr": manifest[sr_key],
        "chunk_suffix": suffix,
        "label": args.label,
        "laws": laws,
        "interp": "z_mix = w1(α)·z_A + w2(α)·z_B; ideal = w1(α)·A + w2(α)·B",
        "alphas": alphas,
        "steps": args.steps,
        "num_pairs_per_track": len(PAIRS),
        "num_tracks": len(tracks),
        "num_trials": len(trials),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "trials": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{len(trials)} trials ({len(laws)} laws × {len(trials)//len(laws)} pairs) -> {args.out}")


if __name__ == "__main__":
    main()
