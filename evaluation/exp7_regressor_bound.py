#!/usr/bin/env python3
"""
E7(a): irreducible-MSE floor for the realistic task  z_mix -> z_voice.

Threat being closed (FM_RESULTS_AND_NEXT.md, H.5 / E7): "the regression baseline
may be under-tuned".  The E1/E6b regressor is claimed to estimate the conditional
mean; if it is under-fitting, a stronger regressor might decode on `softnorm` and
the mechanism claim of S4.3 would collapse to a capacity artefact.

This script measures how much MSE room is left for a *better* regressor:
  * it computes the Bayes-linear (Wiener) floors -- the best L2 estimators that
    use only second-order statistics -- with the PCA machinery of
    exp4_null_task_bound.py (whose wien_pca row it reproduces exactly), plus a
    tighter proper low-rank ridge that regresses the FULL voice latent on the
    mixture-PCA coordinates;
  * it scores the trained regressor checkpoints on the *same* val mixtures, with
    the same MSE-per-element convention as train_fm_cond's val loss (and on the
    exact first-640-clip subset that training logged, for reproducibility checks);
  * it scores the mean of K sampling draws of an RF checkpoint (latavg) as an
    independent estimate of the posterior-mean MSE.

Reading the output: if the regressor's val MSE is at or below every computable
linear floor, it is extracting structure beyond second-order statistics, i.e. it
is NOT under-fitting in the MSE sense; combined with E7(b) -- a much larger
regressor whose val MSE and decoded SI-SDR do not move -- the capacity-artefact
explanation is closed.  Latent MSE is reported for the floor argument only;
per the docs it must never be used as a quality proxy (it is anti-correlated
with decoded quality in these latent spaces).

Estimators (all on held-out val mixtures):
  noop                z_mix                                      trivial floor
  wien_ch_k?          per-position 64x64 channel Wiener filter  linear, stationary
  wien_pca_k?         exp4_null_task_bound.py's construction    linear (legacy; matches
                      "subtract estimated interference"          null_bound_*.json)
  ridge_pca_k?        ridge of the full voice latent on the    linear, tightest floor
                      k mixture-PCA coordinates (lam swept)
  regressor:<name>    a trained CondLatentDiT (objective mse)
  rf_latavg:<name>    mean of K draws of an RF checkpoint       posterior-mean estimate
  rf_seed0:<name>     one draw (for context)

USAGE
    python evaluation/exp7_regressor_bound.py \
        --mixture_dir evaluation/exp4_mixtures/softnorm \
        --regressor_ckpts training/checkpoints/fm_cond/fm_mse_softnorm_best.pt \
                          training/checkpoints/fm_cond/fm_mse_softnorm_final.pt \
        --rf_ckpts training/checkpoints/fm_cond/fm_null_softnorm_final.pt \
        --device cuda:2 --tag softnorm_10s
"""
from __future__ import annotations
import argparse, functools, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train_fm_cond import CondLatentDiT, sample

print = functools.partial(print, flush=True)


def rows_gpu(arr, idx, device):
    """Sorted index array -> float32 GPU tensor (N, D, T)."""
    return torch.from_numpy(np.array(arr[idx], dtype=np.float32)).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture_dir", required=True)
    p.add_argument("--pca_n", type=int, default=8000, help="mixtures used to fit the PCA basis")
    p.add_argument("--fit_n", type=int, default=20000, help="mixtures used to fit the filters")
    p.add_argument("--lat_n", type=int, default=2000, help="val mixtures scored")
    p.add_argument("--k", default="640,2048", help="PCA ranks (comma list)")
    p.add_argument("--ridge", default="1e-4,1e-3,1e-2", help="ridge sweep for ridge_pca floors")
    p.add_argument("--full_wiener", action="store_true",
                   help="ALSO fit the exact full-DT Bayes-linear Wiener filter "
                        "(DT x DT covariance, chunked; lam tuned on held-out train rows). "
                        "Memory-heavy at 10 s (~30 GB): run on an empty GPU.")
    p.add_argument("--full_ridge", default="1e-4,1e-3,1e-2,1e-1")
    p.add_argument("--tune_n", type=int, default=4000,
                   help="held-out TRAIN rows used to tune the full-wiener ridge "
                        "(selection must not touch the reported val rows)")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit_chunk", type=int, default=2000)
    p.add_argument("--regressor_ckpts", nargs="*", default=[])
    p.add_argument("--rf_ckpts", nargs="*", default=[])
    p.add_argument("--rf_draws", type=int, default=16)
    p.add_argument("--rf_steps", type=int, default=32)
    p.add_argument("--rf_n", type=int, default=0, help="val clips for RF rows (0 = lat_n)")
    p.add_argument("--trainval_subset", type=int, default=640,
                   help="also score trained models on the first-N val clips -- the "
                        "exact subset train_fm_cond's val loss used (20 batches x 32)")
    p.add_argument("--output_dir", default="evaluation/exp7_regression")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    mdir = Path(args.mixture_dir)
    meta = json.loads((mdir / "meta.json").read_text())
    f_tr, f_va = meta["splits"]["train"]["files"], meta["splits"]["val"]["files"]
    tr_mix = np.load(f_tr["zmix"], mmap_mode="r")
    tr_tgt = np.load(f_tr["zvoice"], mmap_mode="r")
    va_mix = np.load(f_va["zmix"], mmap_mode="r")
    va_tgt = np.load(f_va["zvoice"], mmap_mode="r")
    n_tr, D, T = tr_mix.shape
    DT = D * T
    k_list = [int(x) for x in args.k.split(",")]
    lam_list = [float(x) for x in args.ridge.split(",")]
    tag = args.tag or f"{mdir.name}_{'2s' if mdir.parent.name.endswith('_2s') else '10s'}"
    lat_n = min(args.lat_n, len(va_mix))

    Zm = torch.from_numpy(np.array(va_mix[:lat_n], np.float32)).to(device)   # (n,D,T)
    Zv = torch.from_numpy(np.array(va_tgt[:lat_n], np.float32)).to(device)
    ZmF = Zm.reshape(lat_n, -1)                                              # (n,DT)
    energy = Zv.pow(2).mean().item()
    print(f"\n[{tag}] {mdir}  latent ({D},{T})  train {n_tr}  val {lat_n}  "
          f"target energy/element {energy:.4f}")

    results = {"mixture_dir": str(mdir), "ae": meta["ae_checkpoint"], "tag": tag,
               "latent_shape": [int(D), int(T)], "lat_n": lat_n,
               "pca_n": args.pca_n, "fit_n": args.fit_n, "seed": args.seed,
               "target_energy_per_element": energy,
               "config": {k: v for k, v in vars(args).items()
                          if k not in ("mixture_dir",)},
               "estimators": {}}

    def add_row(name, z_hat, extra=None):
        zh = z_hat.reshape(len(z_hat), -1) if z_hat.dim() == 3 else z_hat
        ref = Zv.reshape(lat_n, -1)[: len(zh)]
        mse = (zh - ref).pow(2).mean().item()
        rel = ((zh - ref).norm(dim=1) / (ref.norm(dim=1) + 1e-12)).mean().item()
        row = {"mse": mse, "rel_err": rel, "n": int(len(zh))}
        if extra:
            row.update(extra)
        results["estimators"][name] = row
        extra_s = f"   {extra['note']}" if extra and "note" in extra else ""
        print(f"  {name:<40} mse {mse:8.5f}  rel_err {rel:7.4f}  (n={len(zh)}){extra_s}")
        return row

    add_row("noop", Zm)

    # ---- PCA basis of the centred mixture latents, one per rank ----
    # rng usage mirrors exp4_null_task_bound.py exactly (seed 0) so the legacy
    # wien_pca row reproduces the numbers in null_bound_*.json
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_tr, size=min(args.pca_n, n_tr), replace=False); idx.sort()
    Zp = rows_gpu(tr_mix, idx, device).reshape(len(idx), -1)     # (pca_n, DT)
    mu = Zp.mean(0, keepdim=True)
    Zc = Zp - mu
    del Zp
    bases, capture = {}, {}
    for k in k_list:
        U, S, V = torch.svd_lowrank(Zc, q=k, niter=4)
        bases[k] = V[:, :k].contiguous()                        # (DT, k)
        capture[k] = (S[:k].pow(2).sum() / Zc.pow(2).sum()).item()
        print(f"  [basis k={k}] captured energy {capture[k]*100:.2f}%")
    del Zc, U, S, V
    torch.cuda.empty_cache()
    results["pca_captured_energy"] = {str(k): capture[k] for k in k_list}

    # ---- fit all linear filters on train mixtures, chunked ----
    # rng order mirrors exp4_null_task_bound.py (pca rows, then fit rows) so the
    # legacy wien_pca row reproduces null_bound_*.json exactly.
    fi = rng.choice(n_tr, size=min(args.fit_n, n_tr), replace=False); fi.sort()
    n_f = len(fi)
    Cmm = torch.zeros(D, D, device=device)                     # per-position channel
    Cvm = torch.zeros(D, D, device=device)
    fit_state = {k: {"Sxx": torch.zeros(k, k, device=device),   # x^T x, x centred by mixture mean
                     "Sxv": torch.zeros(k, DT, device=device),  # x^T v, both uncentred
                     "Syx": torch.zeros(k, k, device=device),    # legacy: ((v-mu)P)^T x
                     "sx": torch.zeros(k, device=device)} for k in k_list}
    sv = torch.zeros(DT, device=device)                        # sum of uncentred voice rows
    import time as _time
    t0 = _time.time()
    for c0 in range(0, n_f, args.fit_chunk):
        rows = np.array(fi[c0:min(c0 + args.fit_chunk, n_f)])
        Xm = torch.from_numpy(np.array(tr_mix[rows], np.float32)).to(device)  # (m,D,T)
        Xv = torch.from_numpy(np.array(tr_tgt[rows], np.float32)).to(device)
        m = Xm.shape[0]
        Vf = Xv.reshape(m, -1)                                 # (m,DT)
        Cmm += torch.einsum("ndt,net->de", Xm, Xm) / (n_f * T)
        Cvm += torch.einsum("ndt,net->de", Xv, Xm) / (n_f * T)
        sv += Vf.sum(0)
        xmf = Xm.reshape(m, -1) - mu                           # (m,DT) centred by mixture mean
        Vmu = Vf - mu           # legacy construction centres the voice by the MIXTURE mean too
        for k in k_list:
            st = fit_state[k]
            Xmk = xmf @ bases[k]                               # (m,k)
            st["Sxx"] += Xmk.T @ Xmk
            st["Sxv"] += Xmk.T @ Vf
            st["sx"] += Xmk.sum(0)
            st["Syx"] += (Vmu @ bases[k]).T @ Xmk
        del Xm, Xv, xmf, Vmu
    torch.cuda.empty_cache()
    print(f"  [fit] linear filters fitted on {n_f} train mixtures ({_time.time()-t0:.0f}s)")

    A_ch = Cvm @ torch.linalg.inv(Cmm + 1e-6 * torch.eye(D, device=device))
    add_row("wien_ch", torch.einsum("de,bet->bdt", A_ch, Zm))
    floors = [results["estimators"]["wien_ch"]["mse"]]

    x_val = {k: (ZmF - mu) @ bases[k] for k in k_list}         # (lat_n,k)
    for k in k_list:
        st = fit_state[k]
        mu_v = sv / n_f
        xbar = st["sx"] / n_f
        Cxx = (st["Sxx"] - torch.outer(st["sx"], st["sx"]) / n_f) / n_f
        C_xv = (st["Sxv"] - torch.outer(st["sx"], sv) / n_f) / n_f
        # legacy wien_pca (exact exp4_null_task_bound.py construction, fixed ridge)
        Sx_leg = st["Sxx"] / n_f
        lam_leg = 1e-3 * torch.trace(Sx_leg) / k
        W_leg = (st["Syx"] / n_f) @ torch.linalg.inv(
            Sx_leg + lam_leg * torch.eye(k, device=device))
        pca_leg = ZmF - (x_val[k] - x_val[k] @ W_leg.T) @ bases[k].T
        add_row(f"wien_pca_k{k}", pca_leg.reshape(lat_n, D, T),
                {"lambda": float(lam_leg),
                 "note": "legacy exp4_null_task_bound.py construction"})
        floors.append(results["estimators"][f"wien_pca_k{k}"]["mse"])
        # proper low-rank ridge -> FULL voice latent, ridge swept, best kept
        best, sweep = None, {}
        for lam in lam_list:
            A = torch.linalg.solve(Cxx + lam * torch.eye(k, device=device), C_xv)
            a = mu_v - xbar @ A
            mse = ((a + x_val[k] @ A) - Zv.reshape(lat_n, -1)).pow(2).mean().item()
            sweep[f"{lam:g}"] = mse
            if best is None or mse < best["mse"]:
                best = {"mse": mse, "lam": lam, "A": A, "a": a}
        add_row(f"ridge_pca_k{k}", (best["a"] + x_val[k] @ best["A"]).reshape(lat_n, D, T),
                {"lambda": best["lam"], "lambda_sweep": sweep,
                 "note": f"lam={best['lam']:g} (swept)"})
        floors.append(best["mse"])

    # ---- exact Bayes-linear floor: the FULL-DT Wiener filter ----
    # Implemented in DUAL (sample-space) form: the ridge solution
    #   A = (X^T X + n*lam*I)^-1 X^T Y  =  X^T (X X^T + n*lam*I)^-1 Y
    # has the same predictions via  y_hat = mu_v + (x-mu) X^T (K + n*lam*I)^-1 Yc,
    # where K is the n x n Gram matrix (n=fit rows), so every ridge value in the
    # sweep costs one small solve instead of a DT x DT factorisation.  Memory
    # ~13 GB at 10 s instead of ~40 GB for the primal.  lam is tuned on held-out
    # TRAIN rows, never on the reported val rows.
    if args.full_wiener:
        mu_v = sv / n_f
        pool = np.setdiff1d(np.arange(n_tr), np.concatenate([idx, fi]))
        ti = rng.choice(pool, size=min(args.tune_n, len(pool)), replace=False); ti.sort()
        print(f"  [full-wiener] dual ridge: Gram on {n_f} rows, tuning on "
              f"{len(ti)} held-out train rows ...")
        Xf = torch.empty(n_f, DT, device=device)
        Yf = torch.empty(n_f, DT, device=device)
        for c0 in range(0, n_f, args.fit_chunk):
            rows = np.array(fi[c0:min(c0 + args.fit_chunk, n_f)])
            Xf[c0:c0 + len(rows)] = torch.from_numpy(
                np.array(tr_mix[rows], np.float32)).to(device).reshape(len(rows), -1) - mu
            Yf[c0:c0 + len(rows)] = torch.from_numpy(
                np.array(tr_tgt[rows], np.float32)).to(device).reshape(len(rows), -1) - mu_v
        K = Xf @ Xf.T                                        # (n_f, n_f) Gram
        tr_scale = K.diagonal().sum() / (n_f * DT)
        Xt = torch.empty(len(ti), DT, device=device)
        for c0 in range(0, len(ti), args.fit_chunk):
            rows = np.array(ti[c0:min(c0 + args.fit_chunk, len(ti))])
            Xt[c0:c0 + len(rows)] = torch.from_numpy(
                np.array(tr_mix[rows], np.float32)).to(device).reshape(len(rows), -1) - mu
        Kt = Xt @ Xf.T                                       # (n_tune, n_f), lam-independent
        Yt = torch.empty(len(ti), DT, device=device)
        for c0 in range(0, len(ti), args.fit_chunk):
            rows = np.array(ti[c0:min(c0 + args.fit_chunk, len(ti))])
            Yt[c0:c0 + len(rows)] = torch.from_numpy(
                np.array(tr_tgt[rows], np.float32)).to(device).reshape(len(rows), -1) - mu_v
        eye_n = torch.eye(n_f, device=device)
        sweep_f, best_f = {}, None
        for lam in [float(x) for x in args.full_ridge.split(",")]:
            nlam = n_f * lam * tr_scale
            alpha = torch.linalg.solve(K + nlam * eye_n, Yf)  # (n_f, DT)
            tmse = ((Kt @ alpha - Yt).pow(2)).mean().item()
            sweep_f[f"{lam:g}"] = tmse
            if best_f is None or tmse < best_f[0]:
                best_f = (tmse, lam)
            del alpha
            torch.cuda.empty_cache()
        tmse, lam = best_f
        nlam = n_f * lam * tr_scale
        alpha = torch.linalg.solve(K + nlam * eye_n, Yf)
        add_row("full_wiener", (mu_v + (ZmF - mu) @ Xf.T @ alpha).reshape(lat_n, D, T),
                {"lambda": lam, "lambda_scaled_n": float(nlam), "tune_mse": tmse,
                 "lambda_sweep_tune_mse": sweep_f, "tune_n": int(len(ti)),
                 "note": f"lam={lam:g} (tuned on held-out train rows, dual form)"})
        floors.append(results["estimators"]["full_wiener"]["mse"])
        del Xf, Yf, K, Kt, Xt, Yt, alpha, eye_n
        torch.cuda.empty_cache()

    # ---- trained regressor checkpoints ----
    for ck_path in args.regressor_ckpts:
        ck = torch.load(ck_path, map_location="cpu")
        assert ck.get("objective", "mse") == "mse", f"{ck_path}: not an mse regressor"
        assert int(ck.get("n_cond", 0)) == 0 and ck.get("cond", "mix") == "mix"
        model = CondLatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                             depth=ck["depth"], num_heads=ck["num_heads"],
                             n_cond=0).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        with torch.no_grad():
            preds = [model(Zm[i:i + 32], torch.zeros(len(Zm[i:i + 32]), device=device))
                     for i in range(0, lat_n, 32)]
            z_hat = torch.cat(preds)
        name = Path(ck_path).stem
        sub = min(args.trainval_subset, lat_n)
        mse_sub = (z_hat.reshape(lat_n, -1)[:sub] -
                   Zv.reshape(lat_n, -1)[:sub]).pow(2).mean().item()
        add_row(f"regressor:{name}", z_hat, {
            "params_M": sum(q.numel() for q in model.parameters()) / 1e6,
            "epoch": int(ck.get("epoch", -1)), "val_loss_at_save": ck.get("val_loss"),
            "mse_firstN_trainval": mse_sub, "firstN": sub,
            "note": f"epoch {ck.get('epoch')}, logged val_loss {ck.get('val_loss'):.5f}"
                    if ck.get("val_loss") else ""})
        del model
        torch.cuda.empty_cache()

    # ---- RF checkpoints: latavg (posterior-mean estimate) and one draw ----
    for ck_path in args.rf_ckpts:
        ck = torch.load(ck_path, map_location="cpu")
        assert ck.get("objective", "rf") == "rf" and int(ck.get("n_cond", 1)) == 1
        model = CondLatentDiT(channels=ck["channels"], embed_dim=ck["embed_dim"],
                              depth=ck["depth"], num_heads=ck["num_heads"],
                             n_cond=1).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        n_rf = min(args.rf_n or lat_n, lat_n)
        zs = []
        with torch.no_grad():
            for sd in range(args.rf_draws):
                zs.append(torch.cat([
                    sample(model, (Zm[i:i + 32],), args.rf_steps, device, seed=sd)
                    for i in range(0, n_rf, 32)]))
        name = Path(ck_path).stem
        z_mean = torch.stack(zs).mean(0)
        single = torch.stack([(z - Zv[:n_rf]).pow(2).mean() for z in zs]).mean().item()
        add_row(f"rf_latavg:{name}", z_mean, {
            "draws": args.rf_draws, "steps": args.rf_steps,
            "mean_single_draw_mse": single,
            "note": f"{args.rf_draws} draws @ {args.rf_steps} steps"})
        add_row(f"rf_seed0:{name}", zs[0], {
            "draws": 1, "steps": args.rf_steps})
        del model, zs
        torch.cuda.empty_cache()

    # ---- summary: linear floors vs regressors ----
    best_floor = min(floors)
    print(f"\n  [summary {tag}] best linear floor MSE {best_floor:.5f}")
    for nm, row in results["estimators"].items():
        if nm.startswith("regressor:"):
            gap = (best_floor - row["mse"]) / best_floor * 100
            print(f"    {nm:<44} {row['mse']:.5f}  "
                  f"({gap:+.1f}% vs best linear floor)")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    fjson = out / f"e7_bound_{tag}.json"
    fjson.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {fjson}")


if __name__ == "__main__":
    main()
