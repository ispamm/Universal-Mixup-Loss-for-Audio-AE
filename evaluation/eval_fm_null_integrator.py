#!/usr/bin/env python3
"""
Is the "univ degrades with more sampling steps" effect an Euler artefact?

The post-hoc null-task eval shows univ peaking at ~2 Euler steps and then getting
worse with more steps (6.81 dB at 2 steps -> 5.78 dB at 32), while softnorm
improves monotonically to ~8-16 steps.  A converged, near-straight field should
not behave that way, so this compares Euler against midpoint (RK2) integration at
matched step counts.  If RK2 fixes univ's degradation the anomaly was
discretisation; if it persists, it is a property of the learned field and needs a
different explanation.

USAGE
    python evaluation/eval_fm_null_integrator.py --num_clips 128 --device cuda:0
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
from training.train_fm_cond import CondLatentDiT, ae_decode, get as fetch
from evaluation.eval_fm_null_separation import MODELS


@torch.no_grad()
def sample_int(model, conds, steps, device, seed=0, method="euler"):
    B = conds[0].shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(conds[0].shape, device=device, generator=g)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t0 = ts[i]
        dt = ts[i + 1] - ts[i]
        v = model(z, t0.expand(B), *conds)
        if method == "euler":
            z = z + v * dt
        else:                                   # midpoint / RK2
            z_mid = z + v * (dt / 2)
            v2 = model(z_mid, (t0 + dt / 2).expand(B), *conds)
            z = z + v2 * dt
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clips", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", default="2,4,16,32")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="evaluation/exp4_fm_separation")
    args = p.parse_args()

    device = torch.device(args.device)
    steps_list = [int(s) for s in args.steps.split(",")]
    loss_fn = SAMELoss().to(device)
    out = {}

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
        voice = np.load(mdir / "val_voice_wav.npy", mmap_mode="r")
        n = min(args.num_clips, len(va_mix), len(voice))
        ref_all = np.asarray(voice[:n])

        res = {}
        with torch.no_grad():
            for method in ("euler", "rk2"):
                for st in steps_list:
                    sisdr, msl = [], []
                    for i in range(0, n, args.batch_size):
                        sl = slice(i, min(i + args.batch_size, n))
                        zm = fetch(va_mix, sl, device)
                        z = sample_int(model, (zm,), st, device, seed=0, method=method)
                        wav = ae_decode(ae, z, clip_samples).squeeze(1).cpu().numpy()
                        for j in range(len(wav)):
                            sisdr.append(si_sdr_db(ref_all[sl][j], wav[j]))
                            pr = torch.from_numpy(wav[j]).float().view(1, 1, -1).to(device)
                            tg = torch.from_numpy(np.array(ref_all[sl][j], np.float32)).view(1, 1, -1).to(device)
                            msl.append(loss_fn(pr, tg, use_phase=False).item())
                    res[f"{method}_{st}"] = {"sisdr": float(np.mean(sisdr)),
                                             "ms": float(np.mean(msl))}
                    print(f"  [{label}] {method:>5} {st:>3} steps: "
                          f"SI-SDR {np.mean(sisdr):6.2f} dB | MS-STFT {np.mean(msl):.4f}",
                          flush=True)
        out[label] = res

    print(f"\n{'=' * 76}")
    print(f"Integrator comparison (n={args.num_clips} val mixtures, single seed)")
    print(f"{'=' * 76}")
    print(f"{'model':<10}{'method':<8}" + "".join(f"{st:>10}" for st in steps_list))
    print("-" * 76)
    for label in out:
        for method in ("euler", "rk2"):
            row = "".join(f"{out[label][f'{method}_{st}']['sisdr']:>10.2f}" for st in steps_list)
            print(f"{label:<10}{method:<8}{row}")
    print("-" * 76)
    (Path(args.output_dir) / "fm_null_integrator.json").write_text(json.dumps(out, indent=2))
    print(f"Saved: {Path(args.output_dir) / 'fm_null_integrator.json'}")


if __name__ == "__main__":
    main()
