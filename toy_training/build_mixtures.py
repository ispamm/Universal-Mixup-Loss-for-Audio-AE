#!/usr/bin/env python3
"""
Build the conditional-FM separation dataset (Exp 4, Task A).

Task: given a mixture latent z_mix and the interfering stem's latent z_other,
generate the target voice latent z_voice.  The target is defined by the
autoencoder (`encode(voice)`), NOT by algebraic subtraction, so no additivity
assumption is baked in and the same target is well-posed for both AEs.

Mixtures are **raw sums** `mix = voice + other`.  This is deliberate: measured on
these AEs, algebraic separation from a raw sum scores 16.15 dB for `univ`, but
only 9.96 dB if the mixture is peak-normalised to 0.95 and 5.82 dB under RMS
normalisation -- the latent algebra is calibrated to the natural sum, and
renormalising would silently break the oracle baseline the FM is compared
against.  Corpus clips are already peak-normalised individually by the shared
preprocessing pipeline.

Voice stems come from VCTK (`wav48_silence_trimmed`); the interfering stem is
drawn from FSD50K or FMA so that "other" covers both sound events and music.

Latents are pre-encoded once and stored as memory-mappable .npy arrays so
training never needs the autoencoder in the loop:

    <out>/train_zmix.npy    (N, 64, 625) float32
    <out>/train_zother.npy  (N, 64, 625)
    <out>/train_zvoice.npy  (N, 64, 625)
    <out>/val_*.npy         same, for the FM validation split
    <out>/val_voice_wav.npy (K, T) float32 -- raw voice waveforms for the first K
                            val mixtures, so the eval can anchor the metric ladder
                            (no-op / algebraic oracle / AE ceiling) at run time
    <out>/meta.json         pair provenance, energy ratios, latent stats

USAGE
    python toy_training/build_mixtures.py \
        --ae_checkpoint toy_training/checkpoints/transformer_univ_warm_v2_best.pt \
        --out_dir toy_evaluation/exp4_mixtures \
        --num_train 50000 --num_val 2000 --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys, time, functools
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.data import load_stem_clip, collect_validation_stems
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.train_fm_pilot import collect_training_stems

print = functools.partial(print, flush=True)

VOICE_TAG = "wav48"
ANCHOR_CLIPS = 16          # val mixtures whose raw voice wav is stored for eval


def dataset_tag(path: str) -> str:
    low = path.lower()
    if VOICE_TAG in low:
        return "voice"
    if "fsd50k" in low:
        return "fsd50k"
    if "fma" in low:
        return "fma"
    return "other"


@torch.no_grad()
def encode_batch(wrapper, x: torch.Tensor) -> torch.Tensor:
    """(B,1,T) waveform -> (B,D,T_lat) latent."""
    z = wrapper.model.encode(x)
    if wrapper.model_type == "transformer":
        z = z.permute(0, 2, 1)
    return z


def split_stems(stems):
    voices = [s for s in stems if VOICE_TAG in s[0].lower()]
    others = [s for s in stems if VOICE_TAG not in s[0].lower()]
    return voices, others


def build_split(name, voices, others, n_pairs, wrapper, clip_samples, device,
                encode_batch_size, anchor_wav=False, seed=42, progress_every=200):
    """Encode n_pairs mixtures; return dict of memmapped npy paths + stats."""
    if n_pairs > len(voices):
        raise ValueError(f"requested {n_pairs} pairs but only {len(voices)} voice stems")
    rng = np.random.default_rng(seed)

    # maximum diversity: unique voice stems, others cycled if fewer are available
    vi = rng.permutation(len(voices))[:n_pairs]
    if n_pairs <= len(others):
        oi = rng.permutation(len(others))[:n_pairs]
    else:
        base = rng.permutation(len(others))
        reps = int(np.ceil(n_pairs / len(others)))
        oi = np.tile(base, reps)[:n_pairs]

    # determine latent shape from one probe encode
    a0 = load_stem_clip(voices[vi[0]][0], clip_samples, centre=True).squeeze().numpy()
    z0 = wrapper.encode(a0)
    D, T_lat = z0.shape[1], z0.shape[2]
    print(f"[{name}] latent shape ({D}, {T_lat}), {n_pairs} mixtures")

    paths = {}
    for key in ("zmix", "zother", "zvoice"):
        p = out_dir / f"{name}_{key}.npy"
        paths[key] = p
        mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32,
                                       shape=(n_pairs, D, T_lat))
        del mm

    stats = {"energy_ratio": [], "peak_mix": [], "other_tag": [], "pairs": []}
    anchor_wavs = []
    t0 = time.time()

    for start in range(0, n_pairs, encode_batch_size):
        end = min(start + encode_batch_size, n_pairs)
        B = end - start
        v_list, o_list, m_list = [], [], []
        for k in range(start, end):
            v = load_stem_clip(voices[vi[k]][0], clip_samples, centre=True).squeeze().numpy()
            o = load_stem_clip(others[oi[k]][0], clip_samples, centre=True).squeeze().numpy()
            m = (v + o).astype(np.float32)          # raw sum -- see module docstring
            v_list.append(v); o_list.append(o); m_list.append(m)
            stats["energy_ratio"].append(
                float(np.sqrt(np.mean(v ** 2)) / (np.sqrt(np.mean(m ** 2)) + 1e-12)))
            stats["peak_mix"].append(float(np.abs(m).max()))
            stats["other_tag"].append(dataset_tag(others[oi[k]][0]))
            stats["pairs"].append({"voice": os.path.basename(voices[vi[k]][0]),
                                   "other": os.path.basename(others[oi[k]][0]),
                                   "other_tag": dataset_tag(others[oi[k]][0])})
            if anchor_wav and len(anchor_wavs) < ANCHOR_CLIPS:
                anchor_wavs.append(v)

        xv = torch.from_numpy(np.stack(v_list)).unsqueeze(1).to(device)
        xo = torch.from_numpy(np.stack(o_list)).unsqueeze(1).to(device)
        xm = torch.from_numpy(np.stack(m_list)).unsqueeze(1).to(device)
        zv = encode_batch(wrapper, xv).cpu().numpy()
        zo = encode_batch(wrapper, xo).cpu().numpy()
        zm = encode_batch(wrapper, xm).cpu().numpy()

        for key, arr in (("zvoice", zv), ("zother", zo), ("zmix", zm)):
            mm = np.lib.format.open_memmap(paths[key], mode="r+")
            mm[start:end] = arr
            del mm

        if (start // encode_batch_size) % max(1, progress_every // encode_batch_size) == 0:
            el = time.time() - t0
            done = end
            print(f"  [{name}] {done}/{n_pairs}  ({el:.0f}s, "
                  f"{done / max(el, 1e-9):.1f} pairs/s)")

    if anchor_wav:
        np.save(out_dir / f"{name}_voice_wav.npy",
                np.stack(anchor_wavs).astype(np.float32))

    print(f"[{name}] done in {time.time()-t0:.0f}s")
    return paths, stats


def dump_val_voice(mdir: Path, n_dump: int):
    """Recompute the val pairing and (re)write val_voice_wav.npy with n_dump clips.

    The pairing is deterministic given the seed, so this reproduces the exact same
    mixtures as build_split without re-encoding any latents.  The first
    ANCHOR_CLIPS entries are checked against the existing file so a pairing
    mismatch cannot pass silently.
    """
    meta = json.loads((mdir / "meta.json").read_text())
    a = meta["args"]
    stems = collect_validation_stems(sample_rate=a["sample_rate"])
    voices, others = split_stems(stems)
    n_pairs = meta["splits"]["val"]["n"]
    rng = np.random.default_rng(a["seed"] + 1)
    vi = rng.permutation(len(voices))[:n_pairs]
    if n_pairs > len(others):
        raise ValueError("val pairing used the cyclic branch; dump unsupported")
    oi = rng.permutation(len(others))[:n_pairs]                  # same draw as build_split

    clip_samples = int(a["sample_rate"] * a["clip_seconds"])
    n_take = min(n_dump, n_pairs)
    wavs, mix_wavs, oth_wavs = [], [], []
    for k in range(n_take):
        v = load_stem_clip(voices[vi[k]][0], clip_samples, centre=True).squeeze().numpy()
        o = load_stem_clip(others[oi[k]][0], clip_samples, centre=True).squeeze().numpy()
        wavs.append(v); oth_wavs.append(o); mix_wavs.append((v + o).astype(np.float32))

    old_p = mdir / "val_voice_wav.npy"
    if old_p.exists():
        old = np.load(old_p, mmap_mode="r")
        m = min(len(old), len(wavs))
        err = float(np.abs(np.asarray(old[:m]) - np.stack(wavs)[:m]).max())
        print(f"[verify] max abs diff vs existing anchor wavs: {err:.3e} "
              f"({'OK' if err < 1e-6 else 'MISMATCH -- pairing changed!'})")
        if err >= 1e-6:
            raise SystemExit("[FATAL] recomputed pairing does not match the stored "
                             "anchors; refusing to overwrite")
    np.save(old_p, np.stack(wavs).astype(np.float32))
    np.save(mdir / "val_mix_wav.npy", np.stack(mix_wavs).astype(np.float32))
    np.save(mdir / "val_other_wav.npy", np.stack(oth_wavs).astype(np.float32))
    print(f"[dump] wrote {len(wavs)} voice + mixture + other waveforms to {mdir}")


def main():
    global out_dir
    p = argparse.ArgumentParser()
    p.add_argument("--ae_checkpoint", required=True,
                   help="AE used to encode latents (architecture must match the AE "
                        "each FM will use; only the LATENT SCALE matters for storage, "
                        "so both AEs can share one file ONLY if you verify the latent "
                        "distributions match -- by default build one set per AE)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_train", type=int, default=50000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--encode_batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dump_val_voice", type=int, default=0,
                   help="if >0, only recompute and rewrite val_voice_wav.npy with this "
                        "many clips (no latent re-encoding)")
    args = p.parse_args()

    if args.dump_val_voice:
        return dump_val_voice(Path(args.out_dir), args.dump_val_voice)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clip_samples = int(args.sample_rate * args.clip_seconds)

    wrapper = ToyModelWrapper(args.ae_checkpoint, device)
    print(f"[Setup] AE {Path(args.ae_checkpoint).stem} ({wrapper.model_type})")

    print("[Setup] collecting stems")
    train_stems = collect_training_stems(sample_rate=args.sample_rate)
    val_stems = collect_validation_stems(sample_rate=args.sample_rate)
    tr_v, tr_o = split_stems(train_stems)
    va_v, va_o = split_stems(val_stems)
    print(f"[Setup] train: {len(tr_v)} voice / {len(tr_o)} other")
    print(f"[Setup] val  : {len(va_v)} voice / {len(va_o)} other")

    meta = {"ae_checkpoint": args.ae_checkpoint, "args": vars(args), "splits": {}}
    for name, vv, oo, n, seed, anchor in (("train", tr_v, tr_o, args.num_train,
                                           args.seed, False),
                                          ("val", va_v, va_o, args.num_val,
                                           args.seed + 1, True)):
        paths, stats = build_split(name, vv, oo, n, wrapper, clip_samples, device,
                                   args.encode_batch_size, anchor_wav=anchor, seed=seed)
        zm = np.load(paths["zmix"], mmap_mode="r")
        zv = np.load(paths["zvoice"], mmap_mode="r")
        meta["splits"][name] = {
            "n": n,
            "files": {k: str(v) for k, v in paths.items()},
            "latent_shape": [int(zm.shape[1]), int(zm.shape[2])],
            "energy_ratio_mean": float(np.mean(stats["energy_ratio"])),
            "peak_mix_mean": float(np.mean(stats["peak_mix"])),
            "peak_mix_max": float(np.max(stats["peak_mix"])),
            "other_tag_counts": {t: stats["other_tag"].count(t)
                                 for t in sorted(set(stats["other_tag"]))},
            "zmix_mean": float(zm[:200].mean()), "zmix_std": float(zm[:200].std()),
            "zvoice_std": float(zv[:200].std()),
            "pairs": stats["pairs"][:5],
        }
        m = meta["splits"][name]
        print(f"[{name}] energy ratio {m['energy_ratio_mean']:.3f} | "
              f"peak mix mean {m['peak_mix_mean']:.3f} max {m['peak_mix_max']:.3f} | "
              f"other tags {m['other_tag_counts']} | "
              f"latent std zmix {m['zmix_std']:.4f} zvoice {m['zvoice_std']:.4f}")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {out_dir}/meta.json")


if __name__ == "__main__":
    main()
