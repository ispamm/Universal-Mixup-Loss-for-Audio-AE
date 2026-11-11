#!/usr/bin/env python3
"""
Demo-page Section 4: candidate-separation OPTIONS menu.

Generates a listening menu of candidate in-distribution synthetic mixtures
(VCTK voice + FSD50K/FMA interference from the autoencoders' training corpus,
raw sums, 2 s, 44.1 kHz) for the algebraic-separation demo. For each option
you hear: the mixture, the ground-truth voice, and the Universal model's
algebraic separation D(z_mix - z_other), with its SI-SDR.

The current three demo cases (cases.json "separation_synth") are included as
options so they can be kept. After the author picks the 3 to keep, the full
four-model audio is regenerated for the chosen cases.

    .venv/bin/python toy_evaluation/demo_page/generate_separation_options.py \
        --device cuda:1 --n_new 12
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
from toy_training.train_fm_pilot import collect_training_stems  # noqa: E402
from toy_evaluation.exp3_interp_laws import load_adapter  # noqa: E402
from toy_evaluation.separation_compare.separation_metrics import si_sdr  # noqa: E402
from toy_evaluation.demo_page.generate_demo_audio import (  # noqa: E402
    write_wav, common, SR, CLIP_SECONDS, PEAK_GUARD, RMS_GUARD_ABS,
    RMS_GUARD_TOL, DECODE_SEED)

HERE = _THIS_FILE.parent


def w16(x, path, ref_rms=None):
    cap = max(RMS_GUARD_ABS, RMS_GUARD_TOL * ref_rms) if ref_rms else None
    return write_wav(path, x, SR, peak_guard=PEAK_GUARD, rms_guard=cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--n_new", type=int, default=12,
                    help="number of NEW candidate options to add")
    ap.add_argument("--pool", type=int, default=40,
                    help="candidate pool sampled before filtering")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tags", default="fma,fsd50k",
                    help="interference pools to sample from (comma list: fma, fsd50k)")
    ap.add_argument("--liked", default=None,
                    help="'voice_path,other_path,tag' of an already-liked pair to "
                         "include as option 1 (marked on the page)")
    ap.add_argument("--skip_current", action="store_true",
                    help="do not include the current demo cases from cases.json")
    ap.add_argument("--source_pool", default="vctk",
                    choices=["vctk", "fma", "fsd50k"],
                    help="corpus the SOURCE (separation target) is drawn from")
    ap.add_argument("--append", action="store_true",
                    help="append the new candidates to the existing options "
                         "manifest/page instead of replacing them")
    ap.add_argument("--out_dir", default=str(HERE / "options"))
    ap.add_argument("--min_rms", type=float, default=0.03)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cases = json.loads(Path(args.cases).read_text())
    adapter = load_adapter("univ_441", device, cases["models"]["univ"]["ckpt"])
    clip_n = int(SR * CLIP_SECONDS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = collect_training_stems(sample_rate=SR)
    voices = [s for s in stems if "wav48" in s[0].lower()]
    fma = [s for s in stems if "fma" in s[0].lower()]
    fsd = [s for s in stems if "fsd50k" in s[0].lower()]
    print(f"pool: {len(voices)} voices, {len(fma)} fma, {len(fsd)} fsd50k")

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    # ---------- evaluate one (source, other) pair with the univ model ----------
    def evaluate(source_path, other_path, tag, source_tag):
        v = load_stem_clip(source_path, clip_n, centre=True).squeeze().numpy()
        o = load_stem_clip(other_path, clip_n, centre=True).squeeze().numpy()
        v, o = common(v, o)
        if rms(v) < args.min_rms or rms(o) < args.min_rms:
            return None
        mix = (v + o).astype(np.float32)
        with torch.inference_mode():
            z_mix, z_v, z_o = adapter.encode(mix), adapter.encode(v), adapter.encode(o)
            sep = adapter.decode(z_mix - z_o, len(v), seed=DECODE_SEED)
            rec = adapter.decode(z_v, len(v), seed=DECODE_SEED)
        return {"source": source_path, "other": other_path, "tag": tag,
                "source_tag": source_tag,
                "other_name": Path(other_path).name, "source_name": Path(source_path).name,
                "sep_sisdr": float(si_sdr(v, sep)),
                "recon_sisdr": float(si_sdr(v, rec)),
                "v_rms": rms(v), "o_rms": rms(o),
                "_audio": {"mix": mix, "gt": v, "sep": sep}}

    # ---------- liked pair first, then current demo cases ----------
    options = []
    if args.liked:
        v_p, o_p, tag = [p.strip() for p in args.liked.split(",")]
        r = evaluate(v_p, o_p, tag, "vctk")
        if r:
            r["liked"] = True
            options.append(r)
            print(f"[liked ] {tag:7s} {r['other_name'][:40]:42s} "
                  f"sep {r['sep_sisdr']:6.2f} dB")
    if not args.skip_current:
        for c in cases.get("separation_synth", []):
            r = evaluate(c["voice_path"], c["other_path"], c["other_tag"], "vctk")
            if r:
                r["current"] = True
                options.append(r)
                print(f"[current] {c['other_tag']:7s} {r['other_name'][:40]:42s} "
                      f"sep {r['sep_sisdr']:6.2f} dB")

    # ---------- sample and evaluate new candidates ----------
    rng = np.random.default_rng(args.seed)
    sources = {"vctk": voices, "fma": fma, "fsd50k": fsd}[args.source_pool]
    s_idx = rng.permutation(len(sources))
    tags = [t.strip() for t in args.tags.split(",") if t.strip() in ("fma", "fsd50k")]
    tag_cycle = []
    while len(tag_cycle) < args.n_new:
        tag_cycle.extend(tags)
    tag_cycle = tag_cycle[:args.n_new]
    kept = tried = 0
    while kept < args.n_new and tried < args.pool:
        tag = tag_cycle[kept]
        pool = fma if tag == "fma" else fsd
        source = sources[s_idx[tried % len(s_idx)]]
        other = pool[int(rng.integers(len(pool)))]
        tried += 1
        if source[0] == other[0] or source[2] == other[2]:
            continue        # same file or same source track: skip
        r = evaluate(source[0], other[0], tag, args.source_pool)
        if r is None or r["recon_sisdr"] < 15.0:
            continue
        if r["sep_sisdr"] < 12.0:
            continue
        options.append(r)
        kept += 1
        print(f"[new {kept:2d}] {tag:7s} {r['other_name'][:40]:42s} "
              f"sep {r['sep_sisdr']:6.2f} dB (recon {r['recon_sisdr']:5.2f})")

    # ---------- write audio (appending keeps previous options untouched) ----------
    manifest = []
    if args.append and (out_dir / "options_manifest.json").exists():
        manifest = json.loads((out_dir / "options_manifest.json").read_text())
        print(f"[append] keeping {len(manifest)} existing options")
    for r in options:
        idx = len(manifest) + 1
        d = out_dir / f"option_{idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        info = {k: v for k, v in r.items() if not k.startswith("_")}
        info["dir"] = str(d.relative_to(HERE))
        vr = r["v_rms"]
        for key, audio in (("mixture", r["_audio"]["mix"]),
                           ("ground_truth", r["_audio"]["gt"]),
                           ("univ", r["_audio"]["sep"])):
            w16(audio, d / f"{key}.wav", ref_rms=vr if key == "univ" else None)
        manifest.append(info)

    (out_dir / "options_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---------- options.html ----------
    css = """body{font-family:Georgia,serif;margin:0;color:#1a1a2e;background:#fff}
header{background:linear-gradient(135deg,#101a33,#1d2a52);color:#fff;padding:32px 6vw 26px}
header h1{margin:0 0 8px;font-size:24px} header p{color:#cfd8f0;max-width:900px;line-height:1.5}
main{padding:24px 6vw 50px;max-width:1200px;margin:0 auto}
table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border:1px solid #d9dbe6;padding:10px;vertical-align:top;font-size:14px}
th{background:#f5f6fa;font-family:Helvetica;text-align:left}
td.rowhead{background:#fafbfd;min-width:210px}
.sub{color:#777;font-size:12px;font-family:Helvetica}
audio{width:220px;height:32px;display:block}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;background:#eef1f6;
color:#173b63;font-family:Helvetica;font-size:12px}
.badge.best{background:#0f6cbd;color:#fff}
.cur{color:#0f6cbd;font-weight:600}"""

    rows = ""
    tags_txt = " / ".join({"fma": "FMA music", "fsd50k": "FSD50K sound events"}[t]
                          for t in tags) if tags else "FMA / FSD50K"
    src_label = {"vctk": "speech (VCTK)", "fma": "music (FMA)",
                 "fsd50k": "sound event (FSD50K)"}[args.source_pool]
    for i, m in enumerate(manifest, 1):
        cur = '<span class="cur">(current demo case)</span>' if m.get("current") else ""
        liked = '<span class="cur">(your pick)</span>' if m.get("liked") else ""
        tag = "music (FMA)" if m["tag"] == "fma" else "sound event (FSD50K)"
        stag = {"vctk": "speech (VCTK)", "fma": "music (FMA)",
                "fsd50k": "sound event (FSD50K)"}.get(m.get("source_tag", "vctk"))
        sname = m.get("source_name") or m.get("voice_name")
        best = max(x["sep_sisdr"] for x in manifest)
        cls = "best" if m["sep_sisdr"] == best else ""
        rows += f"""<tr><td class="rowhead"><b>Option {i}</b> {cur}{liked}
<div class="sub">source: {stag}<br>interference: {tag}<br>source file: {sname[:44]}<br>interference: {m['other_name'][:44]}</div></td>
<td><audio controls preload="none" src="{m['dir']}/mixture.wav"></audio></td>
<td><audio controls preload="none" src="{m['dir']}/ground_truth.wav"></audio></td>
<td><audio controls preload="none" src="{m['dir']}/univ.wav"></audio>
<div class="sub"><span class="badge {cls}">univ separation: {m['sep_sisdr']:+.2f} dB</span></div></td></tr>"""

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Separation demo — candidate options</title><style>{css}</style></head><body>
<header><h1>Algebraic separation: candidate mixtures (choose 3)</h1>
<p>Each row is a candidate in-distribution synthetic mixture from the training corpus
(source + {tags_txt} interference, natural raw sum; the source is {src_label} unless noted
per row). Listen to the mixture, the ground-truth source, and the Universal model's latent
subtraction D(z_mix &minus; z_other). Pick the <b>three</b> to keep &mdash; SAME-L / Lin-CAE / DAC
will be generated for the chosen ones.</p></header>
<main><table><tr><th>Option</th><th>Mixture</th><th>Source (ground truth)</th>
<th>Universal separation</th></tr>{rows}</table>
<p class="sub">SI-SDR is scale-invariant, computed on exactly the audio above.
Audio peak-normalised to 0.95 where needed.</p></main></body></html>"""

    (HERE / "options.html").write_text(html)
    print(f"\n{len(manifest)} options -> {out_dir} and {HERE / 'options.html'}")


if __name__ == "__main__":
    main()
