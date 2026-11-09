#!/usr/bin/env python3
"""
E9 — is the decoder approximately LINEAR in the latent? (tests the unifying
mechanism of §6b, no FM training required)

Hypothesis.  Cone/universal supervision trains the autoencoder to be additive and
homogeneous over positive weights, which should make the *decoder* approximately
linear in the latent.  Three of our strongest findings would then be one property:

  * algebraic arithmetic works           (Exp 2)   oracle 0.9 dB below ceiling
  * averaging generated codes is valid   (§2.4)    latavg 8.01 vs -2.98
  * the conditional mean decodes         (E1)      regressor 6.14 vs -1.92

If the decoder is linear, linear operations on codes map to the corresponding
audio operations (that *is* additivity), and latent error maps to audio error
through a fixed operator — so an off-target estimate still decodes to something
structured rather than to garbage.

Measured per AE, on the same val latents:

  1. additivity   ||D(a+b) - D(a) - D(b)|| / ||D(a+b)||
     with (a, b) = (z_voice, z_other), whose sum should decode to the true mixture.
     Also reports the mixture fidelity: SI-SDR of D(z_v + z_o) against the true
     mixture waveform.
  2. homogeneity  ||D(alpha z) - alpha D(z)|| / ||D(alpha z)||
  3. nonlinearity of the local linearisation (a second-order finite difference):
        r1(u) = ||D(z + e u) - D(z)||
        r2(u, v) = ||D(z + e u + e v) - D(z + e u) - D(z + e v) + D(z)||
        curvature = r2 / (r1(u) + r1(v))        -> 0 for a linear decoder
     reported at several eps so the growth rate is visible.
  4. directional sensitivity dispersion: CoV and max/median of r1(u)/eps over
     random unit directions.  A linear decoder has a *stable* sensitivity; a
     nonlinear one is insensitive on average but wild in a few directions (which
     is how univ's decoder can be more sensitive on average yet decode better).

USAGE
    python evaluation/exp9_decoder_linearity.py --device cuda:2
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.model_utils import AutoencoderWrapper
from evaluation.eval_recon_sisdr import si_sdr_db

AES = [
    ("univ", "training/checkpoints/transformer_univ_warm_v2_best.pt",
     "evaluation/exp4_mixtures_2s/univ"),
    ("softnorm", "training/checkpoints/transformer_softMax_full_best.pt",
     "evaluation/exp4_mixtures_2s/softnorm"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clips", type=int, default=16)
    p.add_argument("--num_dirs", type=int, default=6)
    p.add_argument("--eps_list", default="0.02,0.05,0.1,0.2")
    p.add_argument("--alpha_list", default="0.5,2.0")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--output_dir", default="evaluation/exp4_fm_separation")
    args = p.parse_args()

    device = torch.device(args.device)
    eps_list = [float(e) for e in args.eps_list.split(",")]
    alphas = [float(a) for a in args.alpha_list.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, aep, mdir in AES:
        mdir = Path(mdir)
        ae = AutoencoderWrapper(aep, device)
        T_lat = json.loads((mdir / "meta.json").read_text())["splits"]["train"]["latent_shape"][1]
        clip = T_lat * 256
        vf = json.loads((mdir / "meta.json").read_text())["splits"]["val"]["files"]
        n = args.num_clips

        def load(path):
            a = np.load(path, mmap_mode="r")
            return torch.from_numpy(np.array(a[:n], np.float32)).to(device)

        zv = load(vf["zvoice"]); zo = load(vf["zother"]); zm = load(vf["zmix"])
        mix_wav = np.asarray(np.load(mdir / "val_mix_wav.npy", mmap_mode="r")[:n])
        print(f"\n[{label}] AE {Path(aep).stem}  latent {tuple(zv.shape[1:])}  n={n}")

        with torch.no_grad():
            def dec(z):
                return ae.model.decode(z.permute(0, 2, 1) if ae.model_type == "transformer" else z,
                                       original_length=clip)

            def norm(t):
                return float(t.flatten(1).norm(dim=1).mean())

            # ---- 1. additivity (decoder-internal) + mixture fidelity ----
            a, b = zv, zo
            dab, da, db = dec(a + b), dec(a), dec(b)
            add_err = float((dab - da - db).flatten(1).norm(dim=1).mean() /
                            (dab.flatten(1).norm(dim=1).mean() + 1e-12))
            mix_wav_t = dec(zm).squeeze(1).cpu().numpy()          # decode of the true mix code
            add_wav = dab.squeeze(1).cpu().numpy()
            fid_add = float(np.mean([si_sdr_db(mix_wav[i], add_wav[i]) for i in range(n)]))
            fid_via_mix = float(np.mean([si_sdr_db(mix_wav[i], mix_wav_t[i]) for i in range(n)]))
            # and against the true (unencoded) mixture waveform
            true_mix = np.asarray(mix_wav)
            fid_add_true = float(np.mean([si_sdr_db(true_mix[i], add_wav[i]) for i in range(n)]))

            # ---- 2. homogeneity ----
            hom = {}
            for al in alphas:
                dl, dz = dec(al * zv), dec(zv)
                hom[str(al)] = float((dl - al * dz).flatten(1).norm(dim=1).mean() /
                                     (dl.flatten(1).norm(dim=1).mean() + 1e-12))

            # ---- 3+4. local linearisation: curvature and sensitivity spread ----
            base = dec(zv)
            scale = zv.flatten(1).norm(dim=1).mean()
            curv, sens = {}, {}
            g = torch.Generator(device=device).manual_seed(0)
            for e in eps_list:
                c_list, s_list = [], []
                for _ in range(args.num_dirs):
                    u = torch.randn(zv.shape, device=zv.device, generator=g)
                    u = u / u.flatten(1).norm(dim=1).view(-1, 1, 1)
                    v = torch.randn(zv.shape, device=zv.device, generator=g)
                    v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1)
                    du = dec(zv + e * scale * u)
                    dv = dec(zv + e * scale * v)
                    duv = dec(zv + e * scale * (u + v))
                    r1u = (du - base).flatten(1).norm(dim=1)
                    r1v = (dv - base).flatten(1).norm(dim=1)
                    r2 = (duv - du - dv + base).flatten(1).norm(dim=1)
                    c_list.append((r2 / (r1u + r1v + 1e-12)).mean().item())
                    s_list.append((r1u / (e * scale)).mean().item())
                curv[f"{e}"] = float(np.mean(c_list))
                s_list = np.array(s_list)
                sens[f"{e}"] = {"mean": float(s_list.mean()),
                                "cov": float(s_list.std() / (s_list.mean() + 1e-12)),
                                "max_over_median": float(s_list.max() / (np.median(s_list) + 1e-12))}

        results[label] = {
            "ae": aep, "additivity_err": add_err,
            "mixture_fidelity_via_zmix": fid_via_mix,
            "mixture_fidelity_via_sum": fid_add,
            "mixture_fidelity_via_sum_true_wav": fid_add_true,
            "homogeneity": hom, "curvature": curv, "sensitivity": sens,
        }
        print(f"  additivity err           {add_err:.4f}")
        print(f"  mixture SI-SDR: decode(z_mix) {fid_via_mix:6.2f} dB | "
              f"decode(z_v+z_o) vs decoded-mix {fid_add:6.2f} dB | vs true mix "
              f"{fid_add_true:6.2f} dB")
        print("  homogeneity err " + "  ".join(f"a={k}:{v:.4f}" for k, v in hom.items()))
        print("  curvature (0 = linear) " + "  ".join(f"e={k}:{v:.4f}" for k, v in curv.items()))
        print("  sensitivity CoV " + "  ".join(
            f"e={k}:{v['cov']:.3f}" for k, v in sens.items()))

    A, B = AES[0][0], AES[1][0]
    print(f"\n{'=' * 84}")
    print("Decoder linearity — hypothesis: the cone latent gives a ~linear decoder")
    print(f"{'=' * 84}")
    print(f"{'quantity':<34}{A:>16}{B:>16}")
    print("-" * 84)
    for k, name in (("additivity_err", "additivity error (lower=linear)"),
                    ("mixture_fidelity_via_sum_true_wav", "decode(z_v+z_o) vs true mix (dB)")):
        print(f"{name:<34}{results[A][k]:>16.4f}{results[B][k]:>16.4f}")
    for e in eps_list:
        print(f"{'curvature eps=' + str(e):<34}"
              f"{results[A]['curvature'][str(e)]:>16.4f}{results[B]['curvature'][str(e)]:>16.4f}")
    for e in eps_list:
        print(f"{'sens CoV eps=' + str(e):<34}"
              f"{results[A]['sensitivity'][str(e)]['cov']:>16.4f}"
              f"{results[B]['sensitivity'][str(e)]['cov']:>16.4f}")
    for al in alphas:
        print(f"{'homogeneity alpha=' + str(al):<34}"
              f"{results[A]['homogeneity'][str(al)]:>16.4f}"
              f"{results[B]['homogeneity'][str(al)]:>16.4f}")
    print("-" * 84)

    (out / "decoder_linearity.json").write_text(json.dumps(
        {"protocol": vars(args), "results": results}, indent=2))
    print(f"Saved: {out / 'decoder_linearity.json'}")


if __name__ == "__main__":
    main()
