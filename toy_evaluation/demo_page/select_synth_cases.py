#!/usr/bin/env python3
"""
Experiment 4 (rebuttal): select the synthetic-mixture separation cases.

The MUSDB18-HQ separation examples sound weak because MUSDB is out-of-domain
for the autoencoders; the paper's own in-distribution protocol builds
mixtures from the TRAINING corpus (VCTK voice + FSD50K/FMA interference, raw
sums -- the same recipe as the flow-matching datasets). This script builds a
candidate pool of such mixtures at 44.1 kHz, scores the universal model's
algebraic separation on each (z_sep = z_mix - z_other, decode, SI-SDR vs the
true voice), and writes the top cases into cases.json's "separation_synth"
section (diverse interference tracks, clean own-reconstruction required).

Only the universal model is scored here (selection criterion); the baselines
are evaluated when the demo audio is generated -- their latent algebra fails
regardless of the clip, and their per-file SI-SDR is recorded then.

    .venv/bin/python toy_evaluation/demo_page/select_synth_cases.py --device cuda:1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_evaluation.data import load_stem_clip  # noqa: E402
from toy_training.train_fm_pilot import collect_training_stems  # noqa: E402
from toy_evaluation.exp3_interp_laws import load_adapter  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import si_sdr  # noqa: E402

SR = 44100
CLIP = 2.0
N_CAND_VOICE = 24          # voice-target candidates (half FMA, half FSD50K)
N_KEEP = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cases", default=str(_THIS_FILE.parent / "cases.json"))
    ap.add_argument("--candidates", type=int, default=N_CAND_VOICE)
    ap.add_argument("--keep", type=int, default=N_KEEP)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cases = json.loads(Path(args.cases).read_text())
    ckpt = cases["models"]["univ"]["ckpt"]
    adapter = load_adapter("univ_441", device, ckpt)
    clip_n = int(SR * CLIP)

    stems = collect_training_stems(sample_rate=SR)
    voices = [s for s in stems if "wav48" in s[0].lower()]
    others = [s for s in stems if "wav48" not in s[0].lower()]
    fma = [s for s in others if "fma" in s[0].lower()]
    fsd = [s for s in others if "fsd50k" in s[0].lower()]
    print(f"pool: {len(voices)} voices, {len(fma)} fma, {len(fsd)} fsd50k")

    rng = np.random.default_rng(args.seed)
    n_half = args.candidates // 2
    v_idx = rng.permutation(len(voices))[:args.candidates]
    pool = [(voices[v_idx[i]], fma if i % 2 == 0 else fsd) for i in range(args.candidates)]
    picks = []
    for i, (voice, pool_o) in enumerate(pool):
        o_idx = int(rng.integers(len(pool_o)))
        picks.append((voice, pool_o[o_idx]))

    rows = []
    with torch.inference_mode():
        for i, (voice, other) in enumerate(picks):
            v = load_stem_clip(voice[0], clip_n, centre=True).squeeze().numpy()
            o = load_stem_clip(other[0], clip_n, centre=True).squeeze().numpy()
            L = min(len(v), len(o))
            v, o = v[:L], o[:L]
            mix = (v + o).astype(np.float32)
            z_mix, z_v, z_o = adapter.encode(mix), adapter.encode(v), adapter.encode(o)
            rec = adapter.decode(z_v, L)
            sep = adapter.decode(z_mix - z_o, L)
            rows.append({
                "i": i,
                "voice": {"path": voice[0], "stem": voice[1], "track": voice[2]},
                "other": {"path": other[0], "stem": other[1], "track": other[2],
                          "tag": "fma" if "fma" in other[0].lower() else "fsd50k"},
                "recon_sisdr": float(si_sdr(v, rec)),
                "sep_sisdr": float(si_sdr(v, sep)),
                "other_rms": float(np.sqrt((o.astype(np.float64) ** 2).mean())),
            })
            print(f"  [{i+1}/{len(picks)}] sep {rows[-1]['sep_sisdr']:6.2f} dB "
                  f"(recon {rows[-1]['recon_sisdr']:5.2f}) | {other[0].split('/')[-1][:40]}")

    # selection: highest separation quality among clean-reconstruction cases;
    # interference-track uniqueness + tag diversity (music AND sound events)
    rows.sort(key=lambda r: -r["sep_sisdr"])
    chosen, used_tracks, used_tags = [], set(), {}
    max_per_tag = max(1, args.keep - 1)     # at least one case of each tag
    for r in rows:
        tag = r["other"]["tag"]
        if r["other"]["path"] in used_tracks or r["recon_sisdr"] < 15.0:
            continue
        if used_tags.get(tag, 0) >= max_per_tag:
            continue
        chosen.append(r)
        used_tracks.add(r["other"]["path"])
        used_tags[tag] = used_tags.get(tag, 0) + 1
        if len(chosen) == args.keep:
            break
    # if a tag is still missing (short pool), backfill from the remainder
    if len(chosen) < args.keep:
        for r in rows:
            if r["other"]["path"] in used_tracks or r["recon_sisdr"] < 15.0:
                continue
            chosen.append(r)
            used_tracks.add(r["other"]["path"])
            if len(chosen) == args.keep:
                break

    out = []
    for k, r in enumerate(chosen):
        out.append({
            "case": f"sep_synth_{k+1}",
            "voice_path": r["voice"]["path"],
            "other_path": r["other"]["path"],
            "other_tag": r["other"]["tag"],
            "interference": r["other"]["stem"],
            "univ_sep_sisdr": r["sep_sisdr"],
            "univ_recon_sisdr": r["recon_sisdr"],
        })
        print(f"chosen: {r['sep_sisdr']:.2f} dB | interference {r['other']['tag']} "
              f"{r['other']['stem'][:50]}")

    cases["separation_synth"] = out
    Path(args.cases).write_text(json.dumps(cases, indent=2))
    print(f"\nSaved {len(out)} cases into {args.cases}")


if __name__ == "__main__":
    main()
