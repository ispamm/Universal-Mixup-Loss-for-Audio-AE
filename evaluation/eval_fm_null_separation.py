#!/usr/bin/env python3
"""
Rigorous post-hoc evaluation of the two conditional FMs on the REALISTIC
separation task (Exp 4): recover the voice from the mixture latent ALONE, with no
oracle access to the other stems.

The training-time eval used 32 val clips, which is too few to separate a ~1 dB
difference from noise.  This script re-evaluates the saved final checkpoints on
many more held-out mixtures with a proper paired test, plus the estimators and
diagnostics the claim actually rests on:

  no-op      decode(z_mix)                          trivial floor
  FM wavavg  mean of decoded waveforms over draws    MMSE estimator (our separator)
  FM seed0   a single noise draw                     what a naive eval would report
  FM latavg  decode of the mean latent               off-manifold stress test: is
                                                     latent interpolation valid?
  FM bestofK per-clip best draw                      sampling-variance reference
  oracle     decode(z_mix - z_other)                 needs the ORACLE (unfair)
  ceiling    decode(z_voice)                         decoder bound

Why latavg matters.  On a cone/additivity-structured latent, positive-weight
combinations of codes are themselves valid codes, so averaging generated latents
should stay on-manifold and still decode.  On a latent with no such structure the
mean drifts off-manifold and decodes badly.  wavavg - latavg therefore measures a
real, structural property rather than an artefact of the metric.

Metrics: SI-SDR (scale invariant) and MS-STFT (spectral, NOT scale invariant), so
both are reported; they can and do disagree.

USAGE
    python evaluation/eval_fm_null_separation.py \
        --num_clips 256 --seeds 6 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.model_utils import AutoencoderWrapper
from training.losses import SAMELoss
from evaluation.eval_recon_sisdr import si_sdr_db
from training.train_fm_cond import CondLatentDiT, sample, ae_decode
from training.train_fm_cond import get as fetch

MODELS = [
    ("univ", "training/checkpoints/fm_cond/fm_null_univ_final.pt",
     "evaluation/exp4_mixtures/univ"),
    ("softnorm", "training/checkpoints/fm_cond/fm_null_softnorm_final.pt",
     "evaluation/exp4_mixtures/softnorm"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clips", type=int, default=256)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=32)
    p.add_argument("--step_sweep", default="1,2,3,4,6,8,12,16,24,32")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="evaluation/exp4_fm_separation")
    p.add_argument("--ckpt_a", default=None, help="override model A checkpoint")
    p.add_argument("--ckpt_b", default=None, help="override model B checkpoint")
    p.add_argument("--tag", default="", help="suffix for the output JSON")
    p.add_argument("--mdir_a", default=None, help="override model A mixture dir")
    p.add_argument("--mdir_b", default=None, help="override model B mixture dir")
    p.add_argument("--label_a", default=None, help="override model A label")
    p.add_argument("--label_b", default=None, help="override model B label")
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    loss_fn = SAMELoss().to(device)
    step_list = [int(s) for s in args.step_sweep.split(",")]

    results, per_clip = {}, {}
    ref_voice = None
    models_spec = list(MODELS)
    if args.ckpt_a or args.mdir_a or args.label_a:
        models_spec[0] = (args.label_a or models_spec[0][0],
                          args.ckpt_a or models_spec[0][1],
                          args.mdir_a or models_spec[0][2])
    if args.ckpt_b or args.mdir_b or args.label_b:
        models_spec[1] = (args.label_b or models_spec[1][0],
                          args.ckpt_b or models_spec[1][1],
                          args.mdir_b or models_spec[1][2])

    for label, ckpt_path, mdir in models_spec:
        mdir = Path(mdir)
        meta = json.loads((mdir / "meta.json").read_text())
        ck = torch.load(ckpt_path, map_location="cpu")
        n_cond = int(ck.get("n_cond", 1))
        objective = ck.get("objective", "rf")
        assert ck.get("cond", "mix") == "mix", f"{label}: checkpoint is not --cond mix"
        model = CondLatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                              depth=ck["depth"], num_heads=ck["num_heads"],
                              n_cond=n_cond).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        ae = AutoencoderWrapper(ck["ae_checkpoint"], device)
        T_lat = ck["t_lat"]; clip_samples = T_lat * 256
        print(f"\n[{label}] {Path(ckpt_path).name} (epoch {ck.get('epoch')}) "
              f"AE {Path(ck['ae_checkpoint']).stem}")

        # guard: the mixture dir must belong to the SAME AE *and* the same chunk
        # length as the checkpoint, otherwise references and decodes differ in
        # length and every metric is silently wrong.
        if Path(meta["ae_checkpoint"]).name != Path(ck["ae_checkpoint"]).name:
            raise SystemExit(f"[FATAL] {label}: checkpoint was trained with "
                             f"{Path(ck['ae_checkpoint']).name} but --mdir points at "
                             f"{Path(meta['ae_checkpoint']).name}. Pass the matching "
                             f"mixture dir via --mdir_a/--mdir_b.")
        vf = meta["splits"]["val"]["files"]
        va_mix = np.load(vf["zmix"], mmap_mode="r")
        va_oth = np.load(vf["zother"], mmap_mode="r")
        va_tgt = np.load(vf["zvoice"], mmap_mode="r")
        voice = np.load(mdir / "val_voice_wav.npy", mmap_mode="r")
        if voice.shape[1] != clip_samples:
            raise SystemExit(f"[FATAL] {label}: mixture dir '{mdir}' holds "
                             f"{voice.shape[1]}-sample reference waveforms but the "
                             f"checkpoint decodes {clip_samples} samples "
                             f"(t_lat={T_lat}). This dir is a different chunk length; "
                             f"pass the matching dir via --mdir_a/--mdir_b.")
        n = min(args.num_clips, len(va_mix), len(voice))
        if ref_voice is None:
            ref_voice = np.asarray(voice[:n])

        acc = {k: {"sisdr": [], "ms": [], "rel_err": []} for k in
               ("noop", "wavavg", "seed0", "latavg", "bestofK", "oracle", "ceiling")}
        sweep = {st: {"sisdr": [], "ms": []} for st in ([] if objective == "mse" else step_list)}

        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                sl = slice(i, min(i + args.batch_size, n))
                zm = fetch(va_mix, sl, device)
                zo = fetch(va_oth, sl, device)
                zv = fetch(va_tgt, sl, device)
                ref = ref_voice[sl]
                B = zm.shape[0]

                def wav_of(z):
                    return ae_decode(ae, z, clip_samples).squeeze(1).cpu().numpy()

                def score(tag, wav, z_for_err=None):
                    for j in range(len(wav)):
                        acc[tag]["sisdr"].append(si_sdr_db(ref[j], wav[j]))
                        pr = torch.from_numpy(wav[j]).float().view(1, 1, -1).to(device)
                        tg = torch.from_numpy(np.array(ref[j], np.float32)).view(1, 1, -1).to(device)
                        acc[tag]["ms"].append(loss_fn(pr, tg, use_phase=False).item())
                    if z_for_err is not None:
                        rel = ((z_for_err - zv).flatten(1).norm(dim=1) /
                               (zv.flatten(1).norm(dim=1) + 1e-12))
                        acc[tag]["rel_err"] += rel.cpu().tolist()

                score("noop", wav_of(zm), zm)
                score("oracle", wav_of(zm - zo), zm - zo)
                score("ceiling", wav_of(zv), zv)

                zs, ws = [], []
                n_seeds = 1 if objective == "mse" else args.seeds
                for sd in range(n_seeds):
                    if objective == "mse":
                        # deterministic regressor: one forward from z_mix
                        z = model(zm, torch.zeros(zm.shape[0], device=device))
                    else:
                        z = sample(model, (zm,), args.sample_steps, device, seed=sd)
                    zs.append(z.cpu()); ws.append(wav_of(z))
                pc = np.stack([[si_sdr_db(ref[j], ws[s][j]) for j in range(B)]
                               for s in range(len(ws))])               # (draws, B)
                wav_mean = np.mean(ws, axis=0)
                for j in range(B):
                    acc["seed0"]["sisdr"].append(float(pc[0, j]))
                    acc["wavavg"]["sisdr"].append(si_sdr_db(ref[j], wav_mean[j]))
                    acc["bestofK"]["sisdr"].append(float(pc[:, j].max()))
                    for tag, w in (("seed0", ws[0][j:j + 1]),
                                   ("wavavg", wav_mean[j:j + 1]),
                                   ("bestofK", ws[int(pc[:, j].argmax())][j:j + 1])):
                        pr = torch.from_numpy(w).float().view(1, 1, -1).to(device)
                        tg = torch.from_numpy(np.array(ref[j], np.float32)).view(1, 1, -1).to(device)
                        acc[tag]["ms"].append(loss_fn(pr, tg, use_phase=False).item())
                z_mean = torch.stack(zs, 0).mean(0).to(device)
                score("latavg", wav_of(z_mean), z_mean)

                for st in ([] if objective == "mse" else step_list):
                    z = sample(model, (zm,), st, device, seed=0)
                    w = wav_of(z)
                    for j in range(B):
                        sweep[st]["sisdr"].append(si_sdr_db(ref[j], w[j]))
                        pr = torch.from_numpy(w[j]).float().view(1, 1, -1).to(device)
                        tg = torch.from_numpy(np.array(ref[j], np.float32)).view(1, 1, -1).to(device)
                        sweep[st]["ms"].append(loss_fn(pr, tg, use_phase=False).item())

        def m(tag, key="sisdr"):
            return float(np.mean(acc[tag][key]))

        print(f"{'estimator':<12}{'SI-SDR dB':>12}{'MS-STFT':>12}{'latent err':>12}")
        for tag in ("ceiling", "oracle", "noop", "seed0", "bestofK", "latavg", "wavavg"):
            re_ = f"{m(tag, 'rel_err'):>12.4f}" if acc[tag]["rel_err"] else f"{'--':>12}"
            print(f"{tag:<12}{m(tag):>12.2f}{m(tag, 'ms'):>12.4f}{re_}")
        if sweep:
            print("steps: " + "  ".join(f"{st}:{np.mean(sweep[st]['sisdr']):.2f}"
                                        for st in step_list))

        results[label] = {tag: {"sisdr": m(tag), "ms": m(tag, "ms"),
                                "rel_err": m(tag, "rel_err") if acc[tag]["rel_err"] else None}
                          for tag in acc}
        results[label]["sweep"] = {st: {"sisdr": float(np.mean(sweep[st]["sisdr"])),
                                        "ms": float(np.mean(sweep[st]["ms"]))}
                                   for st in sweep}
        per_clip[label] = {tag: acc[tag] for tag in acc}

    # ---------------- paired tests ----------------
    from scipy import stats
    A, B = models_spec[0][0], models_spec[1][0]
    print(f"\n{'=' * 96}")
    print(f"REALISTIC TASK: recover voice from z_mix alone  (n={len(per_clip[A]['wavavg']['sisdr'])} "
          f"val mixtures, {args.seeds} noise draws)")
    print(f"{'=' * 96}")
    print(f"{'estimator':<12}{'univ SI-SDR':>14}{'softnorm SI-SDR':>18}"
          f"{'univ MS-STFT':>15}{'softnorm MS-STFT':>18}")
    print("-" * 96)
    for tag in ("ceiling", "oracle", "noop", "wavavg", "seed0", "latavg", "bestofK"):
        print(f"{tag:<12}{results[A][tag]['sisdr']:>14.2f}{results[B][tag]['sisdr']:>18.2f}"
              f"{results[A][tag]['ms']:>15.4f}{results[B][tag]['ms']:>18.4f}")
    print("-" * 96)
    tests = {}
    for tag in ("wavavg", "seed0", "latavg"):
        sa = np.array(per_clip[A][tag]["sisdr"]); sb = np.array(per_clip[B][tag]["sisdr"])
        ma = np.array(per_clip[A][tag]["ms"]); mb = np.array(per_clip[B][tag]["ms"])
        ts, ps = stats.ttest_rel(sa, sb)
        tm, pm = stats.ttest_rel(ma, mb)
        ws, wps = stats.wilcoxon(sa, sb)
        tests[tag] = {"sisdr_univ": float(sa.mean()), "sisdr_softnorm": float(sb.mean()),
                      "sisdr_delta": float(sb.mean() - sa.mean()),
                      "sisdr_p": float(ps), "sisdr_wilcoxon_p": float(wps),
                      "ms_univ": float(ma.mean()), "ms_softnorm": float(mb.mean()),
                      "ms_p": float(pm),
                      "ms_win_univ": int((ma < mb).sum()), "ms_win_softnorm": int((mb < ma).sum())}
        print(f"[{tag}] SI-SDR: univ {sa.mean():.2f} vs softnorm {sb.mean():.2f} "
              f"(delta {sb.mean()-sa.mean():+.2f}, paired p={ps:.2e}, wilcoxon p={wps:.2e})")
        print(f"{'':9} MS-STFT: univ {ma.mean():.4f} vs softnorm {mb.mean():.4f} "
              f"(p={pm:.2e}, MS-STFT wins univ {int((ma<mb).sum())}/"
              f"{int((mb<ma).sum())} softnorm)")
    print("-" * 96)
    for st in (results[A].get("sweep", {}) if results[B].get("sweep") else {}):
        print(f"  steps={st:<3} {A} {results[A]['sweep'][st]['sisdr']:6.2f}  |  "
              f"{B} {results[B]['sweep'][st]['sisdr']:6.2f}")

    (out / f"fm_null_separation{args.tag}.json").write_text(json.dumps(
        {"protocol": vars(args), "results": results, "paired_tests": tests}, indent=2))
    print(f"\nSaved: {out / f'fm_null_separation{args.tag}.json'}")


if __name__ == "__main__":
    main()
