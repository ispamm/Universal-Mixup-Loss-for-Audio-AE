#!/usr/bin/env python3
"""
Experiment 4 (rebuttal, R3 W3 & Q3): generate the demo-page audio.

For ONE model at a time (the four adapters of exp3_interp_laws.py), this
regenerates the audio of the selected demo cases (toy_evaluation/demo_page/
cases.json) and writes 16-bit PCM wavs:

  reconstruction : decode(encode(stem))                       vs ground truth
  interpolation : decode(w1(a)*z_A + w2(a)*z_B) for 9 alphas  vs ideal audio mix
                  (lerp and trig laws, DIFFERENT pairs per law; segments
                  concatenated with 0.2 s gaps)
  separation    : in-distribution synthetic mixtures from the autoencoders'
                  training corpus (VCTK voice + FSD50K/FMA interference, raw
                  sums): decode(z_mix - z_other) vs the true voice, plus each
                  model's own reconstruction of the voice (= ceiling)

Every written file is scored (SI-SDR vs its reference, scale-invariant) so the
page badges describe exactly what is heard. Loudness safety: MODEL outputs
are additionally capped at RMS 0.15 (they are attenuated if louder -- recorded
in the manifest and flagged on the page, since an out-of-control output level
is itself a failure mode); reference files are only peak-guarded at 0.95.

Run once per model, in the right environment (see cases.json "env"):
    .venv/bin/python      toy_evaluation/demo_page/generate_demo_audio.py --model univ     --device cuda:1
    .venv/bin/python      toy_evaluation/demo_page/generate_demo_audio.py --model same_l   --device cuda:1
    cae_env python        toy_evaluation/demo_page/generate_demo_audio.py --model lin_cae  --device cuda:1
    .venv-dac/bin/python  toy_evaluation/demo_page/generate_demo_audio.py --model dac     --device cuda:1
"""
from __future__ import annotations
import argparse, json, sys, wave
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import load_stem_clip  # noqa: E402
from toy_evaluation.exp3_interp_laws import load_adapter, law_weights  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import si_sdr  # noqa: E402

SR = 44100
CLIP_SECONDS = 2.0
PEAK_GUARD = 0.95
RMS_GUARD_ABS = 0.15       # absolute loudness floor for model outputs
RMS_GUARD_TOL = 1.25       # ... or 1.25x the reference's RMS, whichever is larger:
                           # correctly-levelled outputs pass untouched; true
                           # level explosions (e.g. 3-4x the reference) are tamed
DECODE_SEED = 1234


# --------------------------------------------------------------------------- #
#  wav I/O (stdlib only, works in every environment)
# --------------------------------------------------------------------------- #
def write_wav(path: Path, x: np.ndarray, sr: int,
              peak_guard: float = PEAK_GUARD, rms_guard: float | None = None):
    """float32 mono -> 16-bit PCM wav.

    peak_guard: attenuate anything that would clip (applies to references too).
    rms_guard:  if set, additionally cap the RMS (model outputs only) so that
                pathological output levels stay safe to listen to.
    """
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    gain = 1.0
    peak_guarded = False
    loudness_guarded = False
    if peak > peak_guard:
        gain = peak_guard / peak
        peak_guarded = True
    if rms_guard is not None and rms > 0:
        eff = rms * gain
        if eff > rms_guard:
            gain *= rms_guard / eff
            loudness_guarded = True
    y = np.clip(x * gain, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return {"peak": peak, "rms": rms, "gain": gain,
            "peak_guarded": peak_guarded, "loudness_guarded": loudness_guarded}


def common(a: np.ndarray, b: np.ndarray):
    L = min(len(a), len(b))
    return a[:L], b[:L]


def zmix(z_a, z_b, w1, w2):
    if z_a.shape[-1] != z_b.shape[-1]:
        m = min(z_a.shape[-1], z_b.shape[-1])
        z_a, z_b = z_a[..., :m], z_b[..., :m]
    return w1 * z_a + w2 * z_b


def join_with_gap(segments, gap):
    out = []
    for i, s in enumerate(segments):
        if i > 0 and len(gap):
            out.append(gap)
        out.append(s)
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["univ", "same_l", "lin_cae", "dac"])
    ap.add_argument("--cases", default=str(_THIS_FILE.parent / "cases.json"))
    ap.add_argument("--out_dir", default=str(_THIS_FILE.parent / "audio"))
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cases = json.loads(Path(args.cases).read_text())
    spec = cases["models"][args.model]
    clip_n = int(SR * CLIP_SECONDS)

    adapter = load_adapter(spec["adapter"], device, spec.get("ckpt"))
    print(f"[{args.model}] adapter {type(adapter).__name__} | sr {SR} | {device}")

    alphas = cases["interp_protocol"]["alphas"]
    gap = np.zeros(int(cases["interp_protocol"]["gap_seconds"] * SR), dtype=np.float32)
    out_dir = Path(args.out_dir)
    files = {}

    def save(rel, audio, ref=None, model_output=False):
        rms_cap = None
        if model_output:
            if ref is not None:
                ref_rms = float(np.sqrt(np.mean(np.asarray(ref, dtype=np.float64) ** 2)))
                rms_cap = max(RMS_GUARD_ABS, RMS_GUARD_TOL * ref_rms)
            else:
                rms_cap = RMS_GUARD_ABS
        info = write_wav(out_dir / rel, audio, SR, rms_guard=rms_cap)
        if ref is not None:
            r, a = common(ref, audio)
            info["sisdr"] = float(si_sdr(r, a))
        files[rel] = info
        msg = f"  {rel}  ({len(audio)/SR:.1f}s"
        if "sisdr" in info:
            msg += f", SI-SDR {info['sisdr']:6.2f} dB"
        if info["gain"] < 1.0:
            why = "loudness" if info["loudness_guarded"] else "peak"
            msg += f", attenuated x{info['gain']:.3f} ({why})"
        print(msg + ")")

    # ------------------- reconstruction -------------------
    for c in cases["reconstruction"]:
        idx = int(c["track"][:3])
        x = np.load(_REPO_ROOT / "toy_evaluation/separation_compare/chunks"
                    / f"{idx:03d}" / f"{c['stem']}44.npy").astype(np.float32)
        z = adapter.encode(x)
        rec = adapter.decode(z, len(x), seed=DECODE_SEED)
        base = f"reconstruction/{c['case']}"
        save(f"{base}/ground_truth.wav", x)
        save(f"{base}/{args.model}.wav", rec, ref=x, model_output=True)

    # ------------------- interpolation (per-law cases) -------------------
    for law in ("lerp", "trig"):
        for c in cases["interpolation"][law]:
            idx = int(c["track"][:3])
            chunks = _REPO_ROOT / "toy_evaluation/separation_compare/chunks" / f"{idx:03d}"
            a = np.load(chunks / f"{c['stem_a']}44.npy").astype(np.float32)
            b = np.load(chunks / f"{c['stem_b']}44.npy").astype(np.float32)
            a, b = common(a, b)
            z_a, z_b = adapter.encode(a), adapter.encode(b)
            base = f"interpolation/{law}/{c['case']}"
            ideal_seg, model_seg = [], []
            for al in alphas:
                w1, w2 = law_weights(law, float(al))
                ideal_seg.append((w1 * a + w2 * b).astype(np.float32))
                dec = adapter.decode(zmix(z_a, z_b, w1, w2), len(a), seed=DECODE_SEED)
                model_seg.append(dec)
            ideal = join_with_gap(ideal_seg, gap)
            model = join_with_gap(model_seg, gap)
            save(f"{base}/ideal.wav", ideal)
            save(f"{base}/{args.model}.wav", model, ref=ideal, model_output=True)

    # ------------------- separation (synthetic, in-distribution) -------------------
    for c in cases["separation_synth"]:
        v = load_stem_clip(c["voice_path"], clip_n, centre=True).squeeze().numpy()
        o = load_stem_clip(c["other_path"], clip_n, centre=True).squeeze().numpy()
        v, o = common(v, o)
        mix = (v + o).astype(np.float32)          # raw sum, no renormalisation
        z_mix, z_v, z_o = adapter.encode(mix), adapter.encode(v), adapter.encode(o)
        base = f"separation/{c['case']}"
        save(f"{base}/mixture.wav", mix)
        save(f"{base}/ground_truth.wav", v)
        sep = adapter.decode(z_mix - z_o, len(v), seed=DECODE_SEED)
        save(f"{base}/{args.model}.wav", sep, ref=v, model_output=True)
        rec = adapter.decode(z_v, len(v), seed=DECODE_SEED)
        save(f"{base}/{args.model}_ceiling.wav", rec, ref=v, model_output=True)

    # ------------------- manifest -------------------
    man = {"model": args.model, "adapter": spec["adapter"],
           "ckpt": spec.get("ckpt"), "sr": SR, "decode_seed": DECODE_SEED,
           "rms_guard_abs": RMS_GUARD_ABS, "rms_guard_tol": RMS_GUARD_TOL,
           "files": files}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"manifest_{args.model}.json").write_text(json.dumps(man, indent=2))
    print(f"\nSaved: {out_dir}/manifest_{args.model}.json ({len(files)} files)")


if __name__ == "__main__":
    main()
