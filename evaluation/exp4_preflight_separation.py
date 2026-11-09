#!/usr/bin/env python3
"""
Experiment-4 pre-flight: is a conditional-FM stem-removal experiment viable on
these two 16 kHz autoencoders?

Two questions must be answered BEFORE spending two FM trainings:

(1) In-distribution / headroom.  MUSDB18-HQ was not part of the autoencoders'
    training corpus (VCTK + FSD50K + FMA).  If the AEs cannot reconstruct MUSDB
    content, every downstream number is measuring out-of-distribution behaviour.
    So: measure the AE's own reconstruction quality on MUSDB stems and compare it
    against the same AEs on in-distribution data (their own held-out val stems).

(2) Does the univ advantage exist at 16 kHz, and is there room for a model to
    improve on the trivial baselines?  The ladder is:
        no-op        decode(encode(mix))                      <- trivial floor
        algebraic    decode(z_mix - sum(z_others))            <- needs all stems (oracle)
        AE ceiling   decode(encode(target))                   <- decoder bound
    An FM conditioned only on z_mix has to land between no-op and the algebraic
    oracle.  If no-op ~ algebraic ~ ceiling, the task cannot discriminate.

Decoder-free additivity diagnostics (these are the mechanism, computed with no
decoding and no STFT):

    g  = ||z_mix - sum_s z_s|| / ||z_mix||
         how additive the latent is over the full stem decomposition

    h  = ||z_target - (z_mix - sum_{s != target} z_s)|| / ||z_target||
         how close the *algebraic* estimate is to the *encoded* target -- this is
         exactly the gap an FM would have to close

If h is small the algebraic route already solves the task (good for the AE story,
but it means an FM adds little); if h is small for univ and large for softnorm,
the FM experiment is well motivated and univ has structure softnorm lacks.

Suggested reading:
  * recon quality on MUSDB close to in-distribution quality -> MUSDB is usable
  * h(univ) << h(softnorm)                                  -> experiment is worth running
  * algebraic >> no-op                                      -> there is real headroom

USAGE
    python evaluation/exp4_preflight_separation.py \
        --ae_a training/checkpoints/transformer_univ_warm_v2_best.pt \
        --ae_b training/checkpoints/transformer_softMax_full_best.pt \
        --label_a univ --label_b softnorm --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss
from evaluation.data import collect_validation_stems, load_stem_clip
from evaluation.eval_recon_sisdr import si_sdr_db

MUSDB_STEMS = ["drums", "bass", "other", "vocals"]
DATASET_OF = {"wav48": "voice(VCTK)", "fsd50k": "sound(FSD50K)", "fma": "music(FMA)"}


def dataset_tag(pt_path: str) -> str:
    low = pt_path.lower()
    for k, v in DATASET_OF.items():
        if k in low:
            return v
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae_a", required=True)
    p.add_argument("--ae_b", required=True)
    p.add_argument("--label_a", default="univ")
    p.add_argument("--label_b", default="softnorm")
    p.add_argument("--musdb_chunks", default="evaluation/separation_compare/chunks")
    p.add_argument("--num_tracks", type=int, default=100,
                   help="MUSDB tracks to use (max = chunks available)")
    p.add_argument("--num_pairs", type=int, default=100,
                   help="synthetic in-distribution (voice + other) mixtures")
    p.add_argument("--target_stems", default="vocals",
                   help="comma list of MUSDB targets to evaluate (e.g. 'vocals' or 'all')")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="evaluation/exp4_preflight")
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    loss_fn = SAMELoss().to(device)

    models = {}
    for lab, path in [(args.label_a, args.ae_a), (args.label_b, args.ae_b)]:
        models[lab] = AutoencoderWrapper(path, device)
        print(f"[AE] {lab:10s} {Path(path).stem}")

    def msstft(pred, tgt):
        pr = torch.from_numpy(pred).float().view(1, 1, -1).to(device)
        tg = torch.from_numpy(tgt).float().view(1, 1, -1).to(device)
        return loss_fn(pr, tg, use_phase=False).item()

    # =====================================================================
    #  MUSDB: realistic mixtures (possibly out-of-distribution for the AEs)
    # =====================================================================
    man = json.loads((Path(args.musdb_chunks) / "manifest.json").read_text())
    tracks = man["tracks"][: args.num_tracks]
    targets = MUSDB_STEMS if args.target_stems == "all" else args.target_stems.split(",")
    print(f"\n[MUSDB] {len(tracks)} tracks x targets {targets} "
          f"({man['clip_seconds']}s @ {man['sr16']} Hz)")

    mus = {lab: {t: {k: [] for k in ("recon_sisdr", "noop_sisdr", "alg_sisdr",
                                     "recon_ms", "noop_ms", "alg_ms",
                                     "g", "g_s", "h", "h_s", "al_h",
                                     "voice_energy_ratio")}
                 for t in targets} for lab in models}

    for ti, tr in enumerate(tracks):
        d = Path(args.musdb_chunks) / tr["id"].split("__")[0]
        mix = np.load(d / "mix16.npy")
        stems = {s: np.load(d / f"{s}16.npy") for s in MUSDB_STEMS}
        n = len(mix)

        for lab, ae in models.items():
            z_mix = ae.encode(mix)
            z_s = {s: ae.encode(stems[s]) for s in MUSDB_STEMS}
            z_sum = sum(z_s.values())
            g = float((z_mix - z_sum).norm() / (z_mix.norm() + 1e-12))
            # scale-invariant version: SI-SDR is scale invariant, so the relevant
            # question is whether z_sum is *proportional* to z_mix, not equal.
            al_g = float((z_sum * z_mix).sum() / ((z_sum * z_sum).sum() + 1e-12))
            g_s = float((z_mix - al_g * z_sum).norm() / (z_mix.norm() + 1e-12))

            for t in targets:
                z_alg = z_mix - sum(z_s[s] for s in MUSDB_STEMS if s != t)

                z_t = z_s[t]
                rec = ae.decode(z_t, n)
                nop = ae.decode(z_mix, n)
                alg = ae.decode(z_alg, n)
                h = float((z_t - z_alg).norm() / (z_t.norm() + 1e-12))
                al_h = float((z_alg * z_t).sum() / ((z_alg * z_alg).sum() + 1e-12))
                h_s = float((z_t - al_h * z_alg).norm() / (z_t.norm() + 1e-12))

                r = mus[lab][t]
                r["recon_sisdr"].append(si_sdr_db(stems[t], rec))
                r["noop_sisdr"].append(si_sdr_db(stems[t], nop))
                r["alg_sisdr"].append(si_sdr_db(stems[t], alg))
                r["recon_ms"].append(msstft(rec, stems[t]))
                r["noop_ms"].append(msstft(nop, stems[t]))
                r["alg_ms"].append(msstft(alg, stems[t]))
                r["g"].append(g); r["g_s"].append(g_s); r["al_h"].append(al_h)
                r["h"].append(h); r["h_s"].append(h_s)
                r["voice_energy_ratio"].append(
                    float(np.sqrt(np.mean(stems[t] ** 2)) /
                          (np.sqrt(np.mean(mix ** 2)) + 1e-12)))
        if (ti + 1) % 20 == 0:
            print(f"  [musdb {ti+1}/{len(tracks)}]", flush=True)

    # =====================================================================
    #  Synthetic in-distribution mixtures: voice (VCTK) + other sound
    # =====================================================================
    print(f"\n[Synth] building {args.num_pairs} voice+other mixtures from held-out "
          "val stems (same corpus the AEs were trained on)")
    stems_all = collect_validation_stems(sample_rate=16000)
    voices = [s for s in stems_all if "wav48" in s[0].lower()]
    others = [s for s in stems_all if "wav48" not in s[0].lower()]
    rng = np.random.default_rng(args.seed)
    n_pairs = min(args.num_pairs, len(voices), len(others))
    vi = rng.choice(len(voices), size=n_pairs, replace=False)
    oi = rng.choice(len(others), size=n_pairs, replace=False)

    syn = {lab: {k: [] for k in ("recon_sisdr", "noop_sisdr", "alg_sisdr",
                                 "recon_ms", "noop_ms", "alg_ms",
                                 "g", "g_s", "h", "h_s", "al_h",
                                     "voice_energy_ratio")}
           for lab in models}

    for k in range(n_pairs):
        vp, vn, _ = voices[vi[k]]
        op, on, _ = others[oi[k]]
        n = 10 * 16000
        v = load_stem_clip(vp, n, centre=True).squeeze().numpy()
        o = load_stem_clip(op, n, centre=True).squeeze().numpy()
        mix = v + o

        for lab, ae in models.items():
            z_v, z_o, z_m = ae.encode(v), ae.encode(o), ae.encode(mix)
            z_sum = z_v + z_o
            g = float((z_m - z_sum).norm() / (z_m.norm() + 1e-12))
            al_g = float((z_sum * z_m).sum() / ((z_sum * z_sum).sum() + 1e-12))
            g_s = float((z_m - al_g * z_sum).norm() / (z_m.norm() + 1e-12))
            z_alg = z_m - z_o
            h = float((z_v - z_alg).norm() / (z_v.norm() + 1e-12))
            al_h = float((z_alg * z_v).sum() / ((z_alg * z_alg).sum() + 1e-12))
            h_s = float((z_v - al_h * z_alg).norm() / (z_v.norm() + 1e-12))

            rec = ae.decode(z_v, n)
            nop = ae.decode(z_m, n)
            alg = ae.decode(z_alg, n)
            r = syn[lab]
            r["recon_sisdr"].append(si_sdr_db(v, rec))
            r["noop_sisdr"].append(si_sdr_db(v, nop))
            r["alg_sisdr"].append(si_sdr_db(v, alg))
            r["recon_ms"].append(msstft(rec, v))
            r["noop_ms"].append(msstft(nop, v))
            r["alg_ms"].append(msstft(alg, v))
            r["g"].append(g); r["g_s"].append(g_s); r["al_h"].append(al_h)
            r["h"].append(h); r["h_s"].append(h_s)
            r["voice_energy_ratio"].append(
                float(np.sqrt(np.mean(v ** 2)) / (np.sqrt(np.mean(mix ** 2)) + 1e-12)))
        if (k + 1) % 25 == 0:
            print(f"  [synth {k+1}/{n_pairs}]", flush=True)

    # =====================================================================
    #  Report
    # =====================================================================
    def agg(d, lab):
        o = {}
        for k, v in d[lab].items():
            if isinstance(v, dict):
                o[k] = {kk: (float(np.mean(vv)), float(np.std(vv)), len(vv))
                        for kk, vv in v.items() if len(vv)}
            else:
                o[k] = (float(np.mean(v)), float(np.std(v)), len(v))
        return o

    report = {"musdb": {}, "synthetic": {}, "protocol": vars(args)}
    W = 104
    for name, data, extra in (("MUSDB (realistic; NOT in AE training corpus)", mus, targets),
                              ("SYNTHETIC voice+other (in AE training corpus)", syn, [])):
        print(f"\n{'=' * W}")
        print(name)
        print(f"{'=' * W}")
        for lab in models:
            if extra and name.startswith("MUSDB"):
                rows = {t: agg(data, lab)[t] for t in extra}
            else:
                rows = {"voice": agg(data, lab)}
            report["musdb" if name.startswith("MUSDB") else "synthetic"][lab] = (
                {t: {k: list(v) for k, v in r.items()} for t, r in rows.items()})
            for t, r in rows.items():
                print(f"\n  [{lab}] target = {t}   (n={r['recon_sisdr'][2]})")
                print(f"    recon (AE ceiling) SI-SDR {r['recon_sisdr'][0]:7.2f} dB | "
                      f"MS-STFT {r['recon_ms'][0]:8.5f}")
                print(f"    no-op  (decode z_mix)   {r['noop_sisdr'][0]:7.2f} dB | "
                      f"MS-STFT {r['noop_ms'][0]:8.5f}")
                print(f"    algebraic sep           {r['alg_sisdr'][0]:7.2f} dB | "
                      f"MS-STFT {r['alg_ms'][0]:8.5f}")
                d_s = r['alg_sisdr'][0] - r['noop_sisdr'][0]
                d_c = r['recon_sisdr'][0] - r['alg_sisdr'][0]
                print(f"      -> algebraic beats no-op by {d_s:+6.2f} dB ; "
                      f"gap to AE ceiling {d_c:6.2f} dB   (headroom for an FM)")
                print(f"    decoder-free: g={r['g'][0]:.4f} (scale-inv {r['g_s'][0]:.4f}) | "
                      f"h={r['h'][0]:.4f} (scale-inv {r['h_s'][0]:.4f}, "
                      f"alpha={r['al_h'][0]:.3f})")
                print(f"    target/mix energy ratio {r['voice_energy_ratio'][0]:.3f}")

    # head-to-head
    print(f"\n{'=' * W}")
    print("HEAD-TO-HEAD (decoder-free, lower = more additive)")
    print(f"{'=' * W}")
    for name, data in (("MUSDB", mus), ("SYNTHETIC", syn)):
        print(f"  {name}:")
        for lab in models:
            a = agg(data, lab)
            rows = {"voice": a} if "recon_sisdr" in a else a
            for t, r in rows.items():
                print(f"    {lab:10s} [{t:6s}] g={r['g'][0]:.4f} (scale-inv "
                      f"{r['g_s'][0]:.4f})  h={r['h'][0]:.4f} (scale-inv {r['h_s'][0]:.4f}, "
                      f"alpha={r['al_h'][0]:.3f})")
                print(f"    {'':10s}  alg={r['alg_sisdr'][0]:6.2f} dB  "
                      f"no-op={r['noop_sisdr'][0]:6.2f} dB  "
                      f"recon={r['recon_sisdr'][0]:6.2f} dB")
    ga, gb = agg(mus, args.label_a), agg(mus, args.label_b)
    ta = targets[0]
    print(f"\n  VERDICT inputs:")
    print(f"    MUSDB vocals: univ alg {ga[ta]['alg_sisdr'][0]:.2f} dB vs "
          f"softnorm {gb[ta]['alg_sisdr'][0]:.2f} dB")
    print(f"    MUSDB vocals: univ h {ga[ta]['h'][0]:.4f} vs softnorm {gb[ta]['h'][0]:.4f}")
    sa, sb = agg(syn, args.label_a), agg(syn, args.label_b)
    print(f"    SYNTH voice : univ alg {sa['alg_sisdr'][0]:.2f} dB vs "
          f"softnorm {sb['alg_sisdr'][0]:.2f} dB")
    print(f"    SYNTH voice : univ h {sa['h'][0]:.4f} vs softnorm {sb['h'][0]:.4f}")

    (out / "preflight.json").write_text(json.dumps(report, indent=2))
    print(f"\nJSON: {out / 'preflight.json'}")


if __name__ == "__main__":
    main()
