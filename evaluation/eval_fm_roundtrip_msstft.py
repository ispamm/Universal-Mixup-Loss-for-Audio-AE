#!/usr/bin/env python3
"""
Flow-matching round-trip reconstruction quality, measured with MS-STFT.

PROTOCOL
--------
A rectified-flow model has no input to reconstruct: it maps noise -> data.  To
measure a *reconstruction* quality for an FM checkpoint we use the round trip
through its own learned velocity field, in the latent space it was trained on:

    x  --AE.encode-->  z0
    z0 --FM forward-->  z1        integrate dz/dt = v(z, t) from t=0 (data)
                                  to t=1 (noise), same convention as training
                                  (z_t = (1-t)z_data + t*z_noise, v = z_noise - z_data)
    z1 --FM backward--> z_hat     integrate the SAME field from t=1 back to t=0
    z_hat --AE.decode--> x_hat
    metric: MS-STFT(x, x_hat)

The round trip is deterministic (no noise is drawn anywhere), so per-clip scores
are paired across models and a paired test is valid.

Two reference points are reported alongside:
  * AE ceiling -- ae.decode(ae.encode(x)) with no FM in the loop.  This is the
    best the FM round trip could possibly do; the gap between ceiling and
    round trip is the part attributable to the flow model.
  * gain-matched MS-STFT -- MS-STFT after least-squares gain matching of x_hat
    onto x.  The raw MS-STFT is not scale invariant, so if a round-tripped
    latent drifts in energy the raw number partly measures loudness rather than
    spectral fidelity; the gain-matched column removes that confound.

USAGE
-----
    python evaluation/eval_fm_roundtrip_msstft.py \
        --fm_a training/checkpoints/fm_pilot/fm_univ_best.pt \
        --fm_b training/checkpoints/fm_pilot/fm_softnorm_best.pt \
        --label_a universal --label_b softnorm \
        --num_clips 200 --steps 100 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data import collect_validation_stems, load_stem_clip
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss
from training.sample_fm import LatentDiT


# ----------------------------------------------------------------------------
#  AE batched encode / decode
# ----------------------------------------------------------------------------

def load_fm(path, device):
    """Load a LatentDiT checkpoint -> (model, ckpt_dict)."""
    ck = torch.load(path, map_location="cpu")
    model = LatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                      depth=ck["depth"], num_heads=ck["num_heads"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def ae_encode(wrapper, x_batch: torch.Tensor) -> torch.Tensor:
    """(B, 1, T) waveform -> latent (B, D, T_lat)."""
    z = wrapper.model.encode(x_batch)
    if wrapper.model_type == "transformer":
        z = z.permute(0, 2, 1)          # (B, T, D) -> (B, D, T)
    return z


@torch.no_grad()
def ae_decode(wrapper, z: torch.Tensor, orig_len: int) -> torch.Tensor:
    """Latent (B, D, T_lat) -> waveform (B, 1, T)."""
    z_in = z
    if wrapper.model_type == "transformer":
        z_in = z.permute(0, 2, 1)       # (B, D, T) -> (B, T, D)
    return wrapper.model.decode(z_in, original_length=orig_len)


# ----------------------------------------------------------------------------
#  Rectified-flow integration
# ----------------------------------------------------------------------------

@torch.no_grad()
def integrate(model, z, t_start: float, t_end: float, steps: int) -> torch.Tensor:
    """Euler integration of dz/dt = v(z,t) from t_start to t_end (either direction)."""
    B = z.shape[0]
    ts = torch.linspace(t_start, t_end, steps + 1, device=z.device)
    for i in range(steps):
        t = ts[i].expand(B)
        dt = (ts[i + 1] - ts[i])
        z = z + model(z, t) * dt
    return z


def roundtrip(model, z0, steps: int) -> torch.Tensor:
    """data latent -> noise -> data latent, through the learned field."""
    z1 = integrate(model, z0, 0.0, 1.0, steps)
    return integrate(model, z1, 1.0, 0.0, steps), z1


# ----------------------------------------------------------------------------
#  Metrics
# ----------------------------------------------------------------------------

def msstft(loss_fn, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-item MS-STFT (magnitude terms, summed over 7 FFT scales).

    SAMELoss returns a scalar averaged over the batch, so items are scored one at
    a time here to keep the per-clip values needed for a paired test.
    (B,1,T) -> (B,)
    """
    vals = [loss_fn(pred[i:i + 1], target[i:i + 1], use_phase=False).detach()
            for i in range(pred.shape[0])]
    return torch.stack(vals)


def gain_match(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Least-squares optimal scalar gain so pred matches target in energy."""
    p = pred.flatten(1)
    t = target.flatten(1)
    num = (p * t).sum(1, keepdim=True)
    den = (p * p).sum(1, keepdim=True) + 1e-12
    return (num / den).view(-1, 1, 1) * pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_a", required=True)
    p.add_argument("--fm_b", required=True)
    p.add_argument("--label_a", default="universal")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--num_clips", type=int, default=200)
    p.add_argument("--steps", type=int, default=100,
                   help="Euler steps per integration direction (2x this per round trip)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_audio", type=int, default=6,
                   help="clips to write to disk for listening")
    p.add_argument("--output_dir", default="evaluation/eval_fm_roundtrip")
    args = p.parse_args()

    device = torch.device(args.device)
    sr = args.sample_rate
    n = int(sr * args.clip_seconds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"; audio_dir.mkdir(parents=True, exist_ok=True)

    # ---- models ----
    models = {}
    for label, path in [(args.label_a, args.fm_a), (args.label_b, args.fm_b)]:
        fm, ck = load_fm(path, device)
        ae = AutoencoderWrapper(ck["ae_checkpoint"], device)
        models[label] = {"fm": fm, "ck": ck, "ae": ae, "path": str(path)}
        print(f"[Load] {label:12s} FM {Path(path).name} (epoch {ck.get('epoch')}, "
              f"val {ck.get('val_loss'):.4f})  AE {Path(ck['ae_checkpoint']).stem} "
              f"[{ae.model_type}]")

    # ---- data (same held-out validation split as every other eval) ----
    print(f"[Data] collecting validation stems (sr={sr})")
    stems = collect_validation_stems(sample_rate=sr)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(stems), size=min(args.num_clips, len(stems)), replace=False)
    print(f"[Data] {len(stems)} validation stems; using {len(idx)} clips "
          f"(seed {args.seed}, {args.clip_seconds}s each)")

    loss_fn = SAMELoss().to(device)

    labels = [args.label_a, args.label_b]
    res = {lab: {"msstft": [], "msstft_gm": [], "rms_ratio": [],
                 "latent_rel_err": [], "peak": []} for lab in labels}
    ceil = {lab: {"msstft": [], "rms_ratio": []} for lab in labels}

    t0 = time.time()
    n_done = 0
    for start in range(0, len(idx), args.batch_size):
        bidx = idx[start:start + args.batch_size]
        B = len(bidx)
        x_list = [load_stem_clip(stems[i][0], n, centre=True).squeeze().numpy()
                  for i in bidx]
        x_np = np.stack(x_list)                                   # (B, T) float32
        x = torch.from_numpy(x_np).unsqueeze(1).to(device)        # (B, 1, T)

        batch_recon = {}
        batch_ceil = {}
        for lab in labels:
            d = models[lab]
            fm, ae = d["fm"], d["ae"]

            z0 = ae_encode(ae, x)                                 # (B, D, T_lat)
            z_hat, z1 = roundtrip(fm, z0, args.steps)              # (B, D, T_lat)

            x_hat = ae_decode(ae, z_hat, n)                        # (B, 1, T)
            x_ceil = ae_decode(ae, z0, n)                          # AE-only ceiling

            m = msstft(loss_fn, x_hat, x)
            m_gm = msstft(loss_fn, gain_match(x_hat, x), x)
            m_ceil = msstft(loss_fn, x_ceil, x)

            def rms(t_):
                return t_.flatten(1).pow(2).mean(1).sqrt()

            rms_ratio = (rms(x_hat) / (rms(x) + 1e-12))
            lat_err = ((z_hat - z0).flatten(1).norm(dim=1) /
                       (z0.flatten(1).norm(dim=1) + 1e-12))

            res[lab]["msstft"] += m.cpu().tolist()
            res[lab]["msstft_gm"] += m_gm.cpu().tolist()
            res[lab]["rms_ratio"] += rms_ratio.cpu().tolist()
            res[lab]["latent_rel_err"] += lat_err.cpu().tolist()
            res[lab]["peak"] += x_hat.flatten(1).abs().max(1).values.cpu().tolist()
            ceil[lab]["msstft"] += m_ceil.cpu().tolist()
            ceil[lab]["rms_ratio"] += (rms(x_ceil) / (rms(x) + 1e-12)).cpu().tolist()
            batch_recon[lab] = x_hat
            batch_ceil[lab] = x_ceil

        # ---- audio for listening (first --num_audio clips only) ----
        for j in range(B):
            gi = start + j
            if gi >= args.num_audio:
                break
            sf.write(audio_dir / f"clip{gi:02d}_orig.wav", x_np[j], sr)
            for lab in labels:
                sf.write(audio_dir / f"clip{gi:02d}_{lab}_recon.wav",
                         batch_recon[lab][j, 0].cpu().numpy(), sr)
                sf.write(audio_dir / f"clip{gi:02d}_{lab}_ceiling.wav",
                         batch_ceil[lab][j, 0].cpu().numpy(), sr)
            # A/B/C/D/E strip, each segment peak-matched to 0.9:
            #   original | <a> AE ceiling | <a> FM round trip | <b> AE ceiling | <b> FM round trip
            segs = [(x[j:j + 1], "orig")]
            for lab in labels:
                segs.append((batch_ceil[lab][j:j + 1], f"{lab}_ceiling"))
                segs.append((batch_recon[lab][j:j + 1], f"{lab}_recon"))
            parts = []
            for t_, _ in segs:
                a_ = t_[0, 0].cpu().numpy()
                pk = float(np.abs(a_).max())
                parts.append((a_ / (pk + 1e-9) * 0.9).astype(np.float32))
                parts.append(np.zeros(int(0.4 * sr), dtype=np.float32))
            sf.write(audio_dir / f"clip{gi:02d}_strip.wav", np.concatenate(parts), sr)

        n_done += B
        print(f"  [{n_done}/{len(idx)}] t={time.time() - t0:.0f}s", flush=True)

    # ---- summary ----
    from scipy import stats

    def arr(lab, key):
        return np.array(res[lab][key], dtype=np.float64)

    summary = {}
    for lab in labels:
        ck = models[lab]["ck"]
        summary[lab] = {
            "fm_checkpoint": models[lab]["path"],
            "ae_checkpoint": ck["ae_checkpoint"],
            "fm_epoch": ck.get("epoch"),
            "fm_val_loss": float(ck.get("val_loss")),
            "msstft_roundtrip_mean": float(arr(lab, "msstft").mean()),
            "msstft_roundtrip_std": float(arr(lab, "msstft").std()),
            "msstft_roundtrip_median": float(np.median(arr(lab, "msstft"))),
            "msstft_gainmatched_mean": float(arr(lab, "msstft_gm").mean()),
            "msstft_gainmatched_std": float(arr(lab, "msstft_gm").std()),
            "msstft_gainmatched_median": float(np.median(arr(lab, "msstft_gm"))),
            "msstft_ae_ceiling_mean": float(np.array(ceil[lab]["msstft"]).mean()),
            "msstft_ae_ceiling_std": float(np.array(ceil[lab]["msstft"]).std()),
            "msstft_ae_ceiling_median": float(np.median(np.array(ceil[lab]["msstft"]))),
            "rms_ratio_mean": float(arr(lab, "rms_ratio").mean()),
            "latent_rel_err_mean": float(arr(lab, "latent_rel_err").mean()),
            "recon_peak_mean": float(arr(lab, "peak").mean()),
            "n_clips": int(len(res[lab]["msstft"])),
        }

    a, b = labels[0], labels[1]
    ma, mb = arr(a, "msstft"), arr(b, "msstft")
    t_stat, p_val = stats.ttest_rel(ma, mb)
    try:
        w_stat, w_p = stats.wilcoxon(ma, mb)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    diff = mb - ma
    summary["paired_test"] = {
        "pair": f"{b} vs {a}",
        "msstft_delta_mean": float(diff.mean()),
        "msstft_delta_std": float(diff.std()),
        "msstft_t": float(t_stat), "msstft_p": float(p_val),
        "msstft_wilcoxon_stat": float(w_stat), "msstft_wilcoxon_p": float(w_p),
        "cohens_d": float(diff.mean() / (diff.std() + 1e-12)),
        "n_clips": int(len(ma)),
        "wins_b": int((diff < 0).sum()), "wins_a": int((diff > 0).sum()),
    }

    report = {
        "protocol": {
            "metric": "MS-STFT (SAMELoss magnitude terms, use_phase=False, "
                      "7 FFT scales 32..2048, hop=n_fft/4, summed)",
            "recon": "flow round trip: AE.encode -> FM t=0->1 -> FM t=1->0 -> AE.decode",
            "steps_per_direction": args.steps,
            "clip_seconds": args.clip_seconds, "sample_rate": sr,
            "num_clips": int(len(ma)), "batch_size": args.batch_size,
            "seed": args.seed, "device": str(device),
            "data_selection": "collect_validation_stems (held-out val split, seed 42)",
            "note": "round trip is deterministic; clips are paired across models",
        },
        "summary": summary,
        "per_clip": {lab: res[lab] for lab in labels},
        "ae_ceiling_per_clip": {lab: ceil[lab] for lab in labels},
    }
    (out / "fm_roundtrip_msstft.json").write_text(json.dumps(report, indent=2))

    # ---- README for the audio dump ----
    (audio_dir / "README.md").write_text(
        "# FM round-trip listening set\n\n"
        f"clips: first {min(args.num_audio, len(ma))} of the {len(ma)} evaluation clips "
        f"(seed {args.seed}); per-clip MS-STFT values are the first entries of the\n"
        "`per_clip` arrays in `../fm_roundtrip_msstft.json`.\n\n"
        "## Files per clip\n\n"
        "| file | content |\n|---|---|\n"
        "| `clipNN_orig.wav` | original clip (held-out validation data) |\n"
        "| `clipNN_<model>_ceiling.wav` | AE-only recon `decode(encode(x))` - the best the "
        "round trip could do |\n"
        "| `clipNN_<model>_recon.wav` | FM round-trip recon "
        "(encode -> FM 0->1 -> FM 1->0 -> decode) |\n"
        "| `clipNN_strip.wav` | A/B/C/D/E strip, each segment peak-matched to 0.9 |\n\n"
        "## Strip segment order (0.4 s silence between segments)\n\n"
        + "".join(f"{k+1}. `{name}`\n" for k, name in enumerate(
            ["orig"] + [f"{lab}_{suf}" for lab in labels
                        for suf in ("ceiling", "recon")]))
        + "\n## Caveat: the AEs reconstruct with inverted polarity\n\n"
        "These AEs output (approximately) `-x`. A 440 Hz sine round-trips to `-sine` "
        "with r = -1.000 and\nMS-STFT 0.067, i.e. the magnitude spectrum is reproduced "
        "almost exactly while the waveform\nsign is flipped. This is a property of the AEs "
        "(their training objective is the phase-blind\nMS-STFT, for which the global sign "
        "is unconstrained), not of the FM checkpoints, and it is\ninaudible. It does mean "
        "waveform-correlation diagnostics must be sign-corrected before they can\n"
        "be interpreted - see `../phase_diagnostics.json`.\n", encoding="utf-8")

    # ---- console report ----
    W = 92
    print(f"\n{'=' * W}")
    print(f"FM round-trip reconstruction quality (MS-STFT, lower = better)")
    print(f"protocol: encode -> FM forward (t=0->1) -> FM backward (t=1->0) -> decode")
    print(f"{args.num_clips} held-out clips x {args.clip_seconds}s @ {sr} Hz, "
          f"{args.steps} Euler steps per direction, seed {args.seed}")
    print(f"{'=' * W}")
    print(f"{'model':<12}{'FM val':>8}{'MS-STFT mean±std':>22}{'median':>9}"
          f"{'gain-mat mean':>15}{'AE ceiling':>13}")
    print("-" * W)
    for lab in labels:
        s = summary[lab]
        print(f"{lab:<12}{s['fm_val_loss']:>8.4f}"
              f"{s['msstft_roundtrip_mean']:>13.4f}±{s['msstft_roundtrip_std']:<8.4f}"
              f"{s['msstft_roundtrip_median']:>9.4f}"
              f"{s['msstft_gainmatched_mean']:>15.4f}"
              f"{s['msstft_ae_ceiling_mean']:>13.4f}")
    print("-" * W)
    for lab in labels:
        s = summary[lab]
        print(f"  {lab}: AE {Path(s['ae_checkpoint']).stem} | "
              f"FM {Path(s['fm_checkpoint']).name} (epoch {s['fm_epoch']})")
    print("-" * W)
    for lab in labels:
        s = summary[lab]
        print(f"  {lab}: rms_ratio {s['rms_ratio_mean']:.3f} | "
              f"latent rel err {s['latent_rel_err_mean']:.3f} | "
              f"recon peak {s['recon_peak_mean']:.3f}")
    pt = summary["paired_test"]
    print(f"\nPaired MS-STFT ({pt['pair']}, n={pt['n_clips']}):")
    print(f"  delta = {pt['msstft_delta_mean']:+.4f} ± {pt['msstft_delta_std']:.4f} "
          f"(negative = {b} better)  t={pt['msstft_t']:.2f}  p={pt['msstft_p']:.3e}")
    print(f"  Wilcoxon p={pt['msstft_wilcoxon_p']:.3e}  Cohen's d={pt['cohens_d']:.2f}  "
          f"per-clip wins: {b} {pt['wins_b']} / {a} {pt['wins_a']}")
    print(f"\nAudio: {audio_dir}/clip*_orig.wav | _*_ceiling.wav | _*_recon.wav | _strip.wav")
    print(f"JSON : {out / 'fm_roundtrip_msstft.json'}")


if __name__ == "__main__":
    main()
