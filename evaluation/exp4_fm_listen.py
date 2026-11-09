#!/usr/bin/env python3
"""
Exp 4 — step x estimator sweep, and a listening set.

Two questions this answers that the earlier evaluation could not:

(1) The rigorous post-hoc eval reported averaged (6-draw) numbers only at 32
    steps, while the step sweep was single-seed.  So "univ at 2 steps WITH
    averaging" was never measured, and the natural question -- which
    (model, step count) pair actually gives the best audio -- could not be
    answered.  This script measures the full grid:

        {univ, softnorm} x {1,2,4,8,16,32} steps x {single draw, mean of K draws}

    for SI-SDR and MS-STFT, with paired statistics across models at matched
    configurations.

(2) It writes audio for a few held-out mixtures so the numbers can be judged by
    ear: the true mixture, the target voice, the AE ceiling, the algebraic
    oracle, and both FMs at a few step counts.

USAGE
    python evaluation/exp4_fm_listen.py --num_clips 128 --listen_clips 6 \
        --seeds 6 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss
from evaluation.eval_recon_sisdr import si_sdr_db
from training.train_fm_cond import CondLatentDiT, sample, ae_decode, get as fetch
from evaluation.eval_fm_null_separation import MODELS

SR = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clips", type=int, default=128)
    p.add_argument("--listen_clips", type=int, default=6)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--steps", default="1,2,4,8,16,32")
    p.add_argument("--listen_steps", default="2,32")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="evaluation/exp4_fm_separation")
    p.add_argument("--audio_only", action="store_true",
                   help="only sample the listen steps and write audio; reuse the "
                        "existing grid JSON instead of recomputing the full grid")
    args = p.parse_args()

    device = torch.device(args.device)
    step_list = [int(s) for s in args.steps.split(",")]
    listen_steps = [int(s) for s in args.listen_steps.split(",")]
    if args.audio_only:
        step_list = list(listen_steps)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    audio = out / "audio"; audio.mkdir(parents=True, exist_ok=True)
    loss_fn = SAMELoss().to(device)

    grid = {}                      # grid[model][steps] = {seed0:..., wavavg:...}
    per_clip = {}                  # per_clip[model][steps][estimator][metric] = list
    listen = {}                    # listen[model][steps] = wav array (clips, T)
    listen_baselines = {}

    for label, ckpt_path, mdir in MODELS:
        mdir = Path(mdir)
        meta = json.loads((mdir / "meta.json").read_text())
        ck = torch.load(ckpt_path, map_location="cpu")
        model = CondLatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                              depth=ck["depth"], num_heads=ck["num_heads"],
                              n_cond=int(ck.get("n_cond", 1))).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        ae = AutoencoderWrapper(ck["ae_checkpoint"], device)
        clip_samples = ck["t_lat"] * 256
        vf = meta["splits"]["val"]["files"]
        va_mix = np.load(vf["zmix"], mmap_mode="r")
        va_oth = np.load(vf["zother"], mmap_mode="r")
        va_tgt = np.load(vf["zvoice"], mmap_mode="r")
        voice_wav = np.load(mdir / "val_voice_wav.npy", mmap_mode="r")
        n = min(args.num_clips, len(va_mix), len(voice_wav))
        ref_all = np.asarray(voice_wav[:n])
        print(f"\n[{label}] {Path(ckpt_path).name} AE {Path(ck['ae_checkpoint']).stem}, n={n}")

        grid[label] = {}
        per_clip[label] = {}

        def wav_of(z):
            return ae_decode(ae, z, clip_samples).squeeze(1).cpu().numpy()

        def metrics(wav, ref, sl):
            s, m = [], []
            for j in range(len(wav)):
                s.append(si_sdr_db(ref[sl][j], wav[j]))
                pr = torch.from_numpy(wav[j]).float().view(1, 1, -1).to(device)
                tg = torch.from_numpy(np.array(ref[sl][j], np.float32)).view(1, 1, -1).to(device)
                m.append(loss_fn(pr, tg, use_phase=False).item())
            return s, m

        with torch.no_grad():
            for st in step_list:
                pc = {"seed0": {"sisdr": [], "ms": []}, "wavavg": {"sisdr": [], "ms": []}}
                for i in range(0, n, args.batch_size):
                    sl = slice(i, min(i + args.batch_size, n))
                    zm = fetch(va_mix, sl, device)
                    ws = [wav_of(sample(model, (zm,), st, device, seed=sd))
                          for sd in range(args.seeds)]
                    wmean = np.mean(ws, axis=0)
                    s0, m0 = metrics(ws[0], ref_all, sl)
                    sa, ma = metrics(wmean, ref_all, sl)
                    pc["seed0"]["sisdr"] += s0; pc["seed0"]["ms"] += m0
                    pc["wavavg"]["sisdr"] += sa; pc["wavavg"]["ms"] += ma
                    if i == 0 and st in listen_steps:
                        listen[(label, st)] = ws[0][:args.listen_clips].copy()
                        listen[(label, f"{st}_avg")] = wmean[:args.listen_clips].copy()
                        if label == MODELS[0][0]:
                            listen_baselines["mix"] = np.asarray(
                                np.load(mdir / "val_mix_wav.npy", mmap_mode="r")
                            )[:args.listen_clips]
                            listen_baselines["voice"] = np.asarray(ref_all[:args.listen_clips])
                            listen_baselines["ceiling"] = wav_of(fetch(va_tgt, sl, device))[:args.listen_clips]
                            listen_baselines["oracle"] = wav_of(
                                fetch(va_mix, sl, device) - fetch(va_oth, sl, device))[:args.listen_clips]
                grid[label][st] = {
                    "seed0_sisdr": float(np.mean(pc["seed0"]["sisdr"])),
                    "seed0_ms": float(np.mean(pc["seed0"]["ms"])),
                    "wavavg_sisdr": float(np.mean(pc["wavavg"]["sisdr"])),
                    "wavavg_ms": float(np.mean(pc["wavavg"]["ms"])),
                }
                per_clip[label][st] = pc
                g = grid[label][st]
                print(f"  {st:>3} steps | seed0 {g['seed0_sisdr']:6.2f} dB / "
                      f"{g['seed0_ms']:7.4f} MS | wavavg {g['wavavg_sisdr']:6.2f} dB / "
                      f"{g['wavavg_ms']:7.4f} MS", flush=True)

    # ---------------- tables ----------------
    A, B = MODELS[0][0], MODELS[1][0]
    print(f"\n{'=' * 92}")
    print(f"Step x estimator grid  (n={args.num_clips} val mixtures, "
          f"averaging over {args.seeds} draws)")
    print(f"{'=' * 92}")
    for metric, key in (("SI-SDR dB", "sisdr"), ("MS-STFT", "ms")):
        print(f"\n  {metric} (lower is better only for MS-STFT)")
        print(f"  {'steps':<8}" + "".join(f"{m:>22}" for m in (f"{A} seed0", f"{A} avg",
                                                               f"{B} seed0", f"{B} avg")))
        for st in step_list:
            ga, gb = grid[A][st], grid[B][st]
            print(f"  {st:<8}" + "".join(f"{v:>22.3f}" for v in
                                          (ga[f"seed0_{key}"], ga[f"wavavg_{key}"],
                                           gb[f"seed0_{key}"], gb[f"wavavg_{key}"])))

    best = max(((m, st, e, grid[m][st][f"{e}_sisdr"]) for m in (A, B)
                for st in step_list for e in ("seed0", "wavavg")), key=lambda r: r[3])
    print(f"\n  BEST (model, steps, estimator) by SI-SDR: {best[0]} @ {best[1]} steps "
          f"({best[2]}) = {best[3]:.2f} dB")
    bestms = min(((m, st, e, grid[m][st][f"{e}_ms"]) for m in (A, B)
                  for st in step_list for e in ("seed0", "wavavg")), key=lambda r: r[3])
    print(f"  BEST by MS-STFT: {bestms[0]} @ {bestms[1]} steps ({bestms[2]}) = "
          f"{bestms[3]:.4f}")

    # ---------------- paired tests at matched configs ----------------
    from scipy import stats
    print(f"\n  Paired tests (same clips), wavavg estimator:")
    tests = {}
    for st in step_list:
        sa = np.array(per_clip[A][st]["wavavg"]["sisdr"])
        sb = np.array(per_clip[B][st]["wavavg"]["sisdr"])
        ma = np.array(per_clip[A][st]["wavavg"]["ms"])
        mb = np.array(per_clip[B][st]["wavavg"]["ms"])
        t, pv = stats.ttest_rel(sa, sb)
        tm, pm = stats.ttest_rel(ma, mb)
        tests[st] = {"sisdr_univ": float(sa.mean()), "sisdr_softnorm": float(sb.mean()),
                     "sisdr_delta_softnorm_minus_univ": float(sb.mean() - sa.mean()),
                     "sisdr_p": float(pv), "ms_univ": float(ma.mean()),
                     "ms_softnorm": float(mb.mean()), "ms_p": float(pm),
                     "ms_wins_univ": int((ma < mb).sum()), "ms_wins_softnorm": int((mb < ma).sum())}
        print(f"    {st:>3} steps: SI-SDR univ {sa.mean():6.2f} vs softnorm {sb.mean():6.2f} "
              f"(d={sb.mean()-sa.mean():+5.2f}, p={pv:.1e}) | MS univ {ma.mean():.4f} vs "
              f"softnorm {mb.mean():.4f} (p={pm:.1e}, wins {int((ma<mb).sum())}/"
              f"{int((mb<ma).sum())})")

    # ---------------- listening set ----------------
    print(f"\n  Writing audio for {args.listen_clips} held-out mixtures...")
    map_lines = []
    K = args.listen_clips
    for i in range(K):
        segs = [("mixture", np.asarray(listen_baselines["mix"][i], np.float32)),
                ("voice_target", listen_baselines["voice"][i]),
                ("AE_ceiling", listen_baselines["ceiling"][i]),
                ("oracle_with_all_stems", listen_baselines["oracle"][i])]
        for lab in (A, B):
            for st in listen_steps:
                segs.append((f"{lab}_{st}steps", listen[(lab, st)][i]))
        segs.append((f"{A}_{listen_steps[-1]}steps_avg",
                     listen[(A, f"{listen_steps[-1]}_avg")][i]))
        segs.append((f"{B}_{listen_steps[-1]}steps_avg",
                     listen[(B, f"{listen_steps[-1]}_avg")][i]))
        if i == 0:
            print("    segment lengths: " + ", ".join(
                f"{n}={np.asarray(w).size}" for n, w in segs))
        for name, w in segs:
            a = np.asarray(w, np.float32)
            if a.size:
                sf.write(audio / f"clip{i:02d}_{name}.wav", a, SR)
        # 5 s centre excerpt strip, each segment peak-matched to 0.9.
        # Compute the excerpt per segment from its own length: segments can differ
        # (a tensor slice shorter than the window would otherwise yield an empty
        # array and blow up on .max()).
        parts = []
        for name, w in segs:
            a = np.asarray(w, np.float32).reshape(-1)
            if a.size == 0:
                print(f"    [warn] clip{i:02d} {name}: EMPTY segment, skipping")
                continue
            c = a.size // 2
            ex = slice(max(0, c - 5 * SR // 2), min(a.size, c + 5 * SR // 2))
            a = a[ex]
            pk = float(np.abs(a).max())
            parts.append((a / (pk if pk > 0 else 1.0) * 0.9).astype(np.float32))
            parts.append(np.zeros(int(0.35 * SR), np.float32))
        strip = np.concatenate(parts)
        sf.write(audio / f"clip{i:02d}_STRIP.wav", strip, SR)
        for k, (name, _) in enumerate(segs):
            t0 = k * (5 + 0.35)
            map_lines.append(f"clip{i:02d}  {t0:6.1f}-{t0+5:6.1f}s  {name}")
        map_lines.append("")

    allparts = []
    for i in range(K):
        a, _ = sf.read(audio / f"clip{i:02d}_STRIP.wav")
        allparts.append(a.astype(np.float32))
        allparts.append(np.zeros(int(1.5 * SR), np.float32))
    cat = np.concatenate(allparts)
    sf.write(out / "LISTEN_ALL.wav", cat, SR)
    (out / "LISTEN_MAP.txt").write_text(
        "LISTEN_ALL.wav - Exp 4 listening set (5 s centre excerpt per segment,\n"
        "segments peak-matched to 0.9, 0.35 s gap within a clip, 1.5 s between clips)\n\n"
        + "\n".join(map_lines) + "\n", encoding="utf-8")
    print(f"    wrote {audio}/clip*_*.wav, {out}/LISTEN_ALL.wav "
          f"({len(cat)/SR:.0f}s), {out}/LISTEN_MAP.txt")

    if args.audio_only:
        print("  (--audio_only: grid JSON left untouched)")
        return
    (out / "fm_step_estimator_grid.json").write_text(json.dumps(
        {"protocol": vars(args), "grid": grid,
         "per_clip": {m: {str(st): per_clip[m][st] for st in step_list} for m in (A, B)},
         "paired_tests": {str(k): v for k, v in tests.items()},
         "best_sisdr": {"model": best[0], "steps": best[1], "estimator": best[2],
                        "value": best[3]},
         "best_msstft": {"model": bestms[0], "steps": bestms[1], "estimator": bestms[2],
                         "value": bestms[3]}}, indent=2))
    print(f"  Saved: {out / 'fm_step_estimator_grid.json'}")


if __name__ == "__main__":
    main()
