#!/usr/bin/env python3
"""
Bound the *realistic* separation task: estimate z_voice from z_mix alone.

This is the task that matters for the paper.  The algebraic oracle
`z_mix - sum(z_others)` needs every other stem's latent at inference, which you
do not have in practice; an FM conditioned only on `z_mix` needs nothing else.
So the question is how much of the voice can be recovered from the mixture latent
alone, and whether the two latent spaces differ in how recoverable it is.

Bayes-linear (Wiener) reference.  With z_mix = z_voice + z_other and independent
sources,

    E[z_voice | z_mix] = Sigma_v (Sigma_v + Sigma_o)^-1 z_mix = Cov(z_voice, z_mix) Cov(z_mix)^-1 z_mix

which is NOT a scalar multiple of z_mix unless the two sources share the same
covariance -- i.e. there IS directional separation information in the second
moments alone.  If the posterior mean were purely radial, scale-invariant SI-SDR
would gain nothing over the no-op baseline; the Wiener filter is the honest test
of whether that objection applies.

Because a full latent covariance is 40000 x 40000, the filter is estimated in a
truncated PCA basis of the centred latents (k components), which is a low-rank
approximation of the same estimator:

    x = (z - mu) P                      project (P = top-k right singular vectors)
    x_v = W x                           W = Sigma_yx (Sigma_x + lambda I)^-1
    z_hat = z_mix - P (x - x_v)         subtract the estimated interference

Reported per AE:
  * no-op           decode(z_mix)                        trivial floor
  * wien-ch         per-position channel filter (64x64, ignores time structure)
  * wien-pca        the low-rank Wiener filter above
  * algebraic oracle  decode(z_mix - z_other)            upper reference (needs oracle)
  * AE ceiling      decode(z_voice)                      decoder bound

plus decoder-free latent relative errors for all of them on many val mixtures.

USAGE
    python toy_evaluation/exp4_null_task_bound.py \
        --mixture_dir toy_evaluation/exp4_mixtures/univ --k 640 --device cuda:3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy_evaluation.model_utils import ToyModelWrapper
from toy_training.losses import SAMELoss
from toy_evaluation.eval_recon_sisdr import si_sdr_db


def ae_decode_batch(ae, z, orig_len):
    """Latent (B,D,T) -> waveform (B,T) numpy."""
    z_in = z.permute(0, 2, 1) if ae.model_type == "transformer" else z
    return ae.model.decode(z_in, original_length=orig_len).squeeze(1).cpu().numpy()


def report(tag, wav, ref, loss_fn, device):
    s = [si_sdr_db(ref[i], wav[i]) for i in range(len(wav))]
    m = []
    for i in range(len(wav)):
        pr = torch.from_numpy(wav[i]).float().view(1, 1, -1).to(device)
        tg = torch.from_numpy(np.array(ref[i], np.float32)).view(1, 1, -1).to(device)
        m.append(loss_fn(pr, tg, use_phase=False).item())
    return float(np.mean(s)), float(np.mean(m))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture_dir", required=True)
    p.add_argument("--k", type=int, default=640, help="PCA components for the Wiener basis")
    p.add_argument("--pca_n", type=int, default=8000, help="mixtures used to fit the basis")
    p.add_argument("--fit_n", type=int, default=20000, help="mixtures used to fit W")
    p.add_argument("--lat_n", type=int, default=2000, help="val mixtures for latent errors")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="toy_evaluation/exp4_preflight")
    args = p.parse_args()

    device = torch.device(args.device)
    mdir = Path(args.mixture_dir)
    meta = json.loads((mdir / "meta.json").read_text())
    f = meta["splits"]["train"]["files"]
    vf = meta["splits"]["val"]["files"]
    ae = ToyModelWrapper(meta["ae_checkpoint"], device)
    loss_fn = SAMELoss().to(device)
    T_lat = meta["splits"]["train"]["latent_shape"][1]
    print(f"[{mdir.name}] AE {Path(meta['ae_checkpoint']).stem}, latent "
          f"({meta['splits']['train']['latent_shape'][0]}, {T_lat})")

    tr_mix = np.load(f["zmix"], mmap_mode="r")
    tr_tgt = np.load(f["zvoice"], mmap_mode="r")
    va_mix = np.load(vf["zmix"], mmap_mode="r")
    va_tgt = np.load(vf["zvoice"], mmap_mode="r")
    va_oth = np.load(vf["zother"], mmap_mode="r")
    D = va_mix.shape[1]
    rng = np.random.default_rng(args.seed)

    # ---- PCA basis of the centred mixture latents ----
    idx = rng.choice(len(tr_mix), size=min(args.pca_n, len(tr_mix)), replace=False)
    idx.sort()
    Zp = torch.from_numpy(np.array(tr_mix[idx], np.float32)).reshape(len(idx), -1).to(device)
    mu = Zp.mean(0, keepdim=True)
    Zc = Zp - mu
    U, S, V = torch.svd_lowrank(Zc, q=args.k, niter=4)
    P = V                                                  # (D*T, k)
    print(f"[basis] k={args.k}, captured energy "
          f"{(S.pow(2).sum() / Zc.pow(2).sum()).item()*100:.2f}%")

    # ---- fit W on training mixtures ----
    fi = rng.choice(len(tr_mix), size=min(args.fit_n, len(tr_mix)), replace=False)
    fi.sort()
    Xm = torch.from_numpy(np.array(tr_mix[fi], np.float32)).reshape(len(fi), -1).to(device) - mu
    Xv = torch.from_numpy(np.array(tr_tgt[fi], np.float32)).reshape(len(fi), -1).to(device) - mu
    Xm_p = Xm @ P
    Xv_p = Xv @ P
    Sx = (Xm_p.T @ Xm_p) / len(fi)
    Syx = (Xv_p.T @ Xm_p) / len(fi)
    lam = args.ridge * torch.trace(Sx) / args.k
    W = Syx @ torch.linalg.inv(Sx + lam * torch.eye(args.k, device=device))
    del Xm, Xv, Xm_p, Xv_p, Zp, Zc
    torch.cuda.empty_cache()
    print(f"[Wiener] W fitted on {len(fi)} mixtures (ridge {lam.item():.3e})")

    # ---- per-position channel filter (64x64, no time structure) ----
    Xm_r = torch.from_numpy(np.array(tr_mix[fi], np.float32)).to(device)   # (N,D,T)
    Xv_r = torch.from_numpy(np.array(tr_tgt[fi], np.float32)).to(device)
    Cmm = torch.einsum("ndt,net->de", Xm_r, Xm_r) / (len(fi) * T_lat)
    Cvm = torch.einsum("ndt,net->de", Xv_r, Xm_r) / (len(fi) * T_lat)
    A_ch = Cvm @ torch.linalg.inv(Cmm + 1e-6 * torch.eye(D, device=device))
    del Xm_r, Xv_r
    torch.cuda.empty_cache()

    # ---- evaluate on val ----
    n = min(args.lat_n, len(va_mix))
    out = {}
    lat = {}

    def lat_err(z_hat, ref):
        return float(((z_hat - ref).flatten(1).norm(dim=1) /
                      (ref.flatten(1).norm(dim=1) + 1e-12)).mean())

    with torch.no_grad():
        rels = {"noop": [], "wien_ch": [], "wien_pca": [], "oracle": [], "ceiling": []}
        for i in range(0, n, 64):
            zm = torch.from_numpy(np.array(va_mix[i:i + 64], np.float32)).to(device)
            zv = torch.from_numpy(np.array(va_tgt[i:i + 64], np.float32)).to(device)
            zo = torch.from_numpy(np.array(va_oth[i:i + 64], np.float32)).to(device)
            zmc = zm.reshape(len(zm), -1) - mu
            x = zmc @ P
            x_v = x @ W.T
            # subtract the estimated interference component in the PCA basis
            pca = (zm.reshape(len(zm), -1) - (x - x_v) @ P.T).reshape(zm.shape)
            ch = torch.einsum("de,bet->bdt", A_ch, zm)
            rels["noop"].append(lat_err(zm, zv))
            rels["wien_ch"].append(lat_err(ch, zv))
            rels["wien_pca"].append(lat_err(pca, zv))
            rels["oracle"].append(lat_err(zm - zo, zv))
            rels["ceiling"].append(lat_err(zv, zv))
            if i == 0:
                K = 16
                voice = np.load(mdir / "val_voice_wav.npy", mmap_mode="r")[:K]
                for key, z in (("noop", zm), ("wien_ch", ch), ("wien_pca", pca),
                               ("oracle", zm - zo), ("ceiling", zv)):
                    wav = ae_decode_batch(ae, z[:K], T_lat * 256)
                    out[key] = report(key, wav, voice[:K], loss_fn, device)
    lat = {k: float(np.mean(v)) for k, v in rels.items()}

    print(f"\n{'=' * 88}")
    print(f"Task: estimate voice latent from z_mix ONLY ({mdir.name}, "
          f"{n} val mixtures, {T_lat} frames)")
    print(f"{'=' * 88}")
    print(f"{'estimator':<26}{'latent rel err':>18}{'SI-SDR (16 clips)':>20}{'MS-STFT':>12}")
    print("-" * 88)
    for key in ("noop", "wien_ch", "wien_pca", "oracle", "ceiling"):
        s, m = out[key] if key in out else (float("nan"), float("nan"))
        print(f"{key:<26}{lat[key]:>18.4f}{s:>20.2f}{m:>12.4f}")
    print("-" * 88)
    gain = out["wien_pca"][0] - out["noop"][0] if "wien_pca" in out else float("nan")
    print(f"Wiener(PCA) gain over no-op: {gain:+.2f} dB SI-SDR  |  "
          f"latent {lat['noop']:.4f} -> {lat['wien_pca']:.4f}")
    print(f"headroom left to the AE ceiling: "
          f"{out['ceiling'][0] - out['wien_pca'][0]:.2f} dB" if "wien_pca" in out else "")

    (Path(args.output_dir) / f"null_bound_{mdir.name}.json").write_text(json.dumps(
        {"mixture_dir": str(mdir), "ae": meta["ae_checkpoint"], "k": args.k,
         "sisdr": {k: v[0] for k, v in out.items()},
         "msstft": {k: v[1] for k, v in out.items()},
         "latent_rel_err": lat, "n_val": n}, indent=2))
    print(f"\nSaved: {Path(args.output_dir) / f'null_bound_{mdir.name}.json'}")


if __name__ == "__main__":
    main()
