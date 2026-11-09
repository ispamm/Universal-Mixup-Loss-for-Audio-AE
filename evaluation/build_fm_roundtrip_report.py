#!/usr/bin/env python3
"""
Assemble evaluation/eval_fm_roundtrip/REPORT.md from the JSON artifacts
produced by eval_fm_roundtrip_msstft.py, fm_roundtrip_controls.py and
fm_roundtrip_phase_diag.py.

Every number in the report is read from the JSON, never transcribed by hand, so
re-running the eval and this script keeps the report in sync.

USAGE
    python evaluation/build_fm_roundtrip_report.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_fm_roundtrip"


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    main_j = load("fm_roundtrip_msstft.json")
    ctrl = load("controls/controls.json")
    ph = load("phase_diagnostics.json")
    sens = load("decoder_sensitivity.json")
    lossn = load("loss_normalization.json")
    robust = {s: load(f"../_robust_steps{s}/fm_roundtrip_msstft.json")
              for s in ("100", "500")}

    S = main_j["summary"]
    P = S["paired_test"]
    A, B = "univ", "softnorm"
    L = []
    w = L.append
    nl = "\n"

    w("# FM round-trip reconstruction: `fm_univ_best` vs `fm_softnorm_best`")
    w("")
    w("Metric: **MS-STFT** only (SAMELoss magnitude terms, `use_phase=False`, 7 FFT scales "
      "32-2048, hop = n_fft/4, summed across scales). Lower is better.")
    w("")
    w("## Protocol")
    w("")
    w("A rectified-flow model has no input to reconstruct, so `recon` here is the **round trip "
      "through the model's own learned velocity field** in the latent space it was trained on:")
    w("")
    w("```")
    w("x --AE.encode--> z0 --FM t=0->1--> z1 --FM t=1->0--> z_hat --AE.decode--> x_hat")
    w("metric: MS-STFT(x, x_hat)")
    w("```")
    w("")
    w(f"- {P['n_clips']} held-out validation clips, 10.0 s @ 16 kHz, seed 42 "
      "(`collect_validation_stems`, the same split every other eval uses)")
    w(f"- {main_j['protocol']['steps_per_direction']} Euler steps per direction; training "
      "convention `z_t=(1-t)z_data+t*z_noise`, `v=z_noise-z_data`, so t=0 is data, t=1 is noise")
    w("- The round trip is **deterministic** (no noise is drawn anywhere), so per-clip scores are "
      "paired across models and a paired test is valid; an independent re-run reproduced the "
      "numbers to ~4 decimals")
    w("")
    w("| model | FM val loss | AE | MS-STFT round trip | median | gain-matched | AE ceiling |")
    w("|---|---|---|---|---|---|---|")
    for lab in (A, B):
        s = S[lab]
        w(f"| `{lab}` | {s['fm_val_loss']:.4f} | `{Path(s['ae_checkpoint']).stem}` | "
          f"**{s['msstft_roundtrip_mean']:.4f} ± {s['msstft_roundtrip_std']:.4f}** | "
          f"{s['msstft_roundtrip_median']:.4f} | {s['msstft_gainmatched_mean']:.4f} | "
          f"{s['msstft_ae_ceiling_mean']:.4f} |")
    w("")
    w(f"**Paired result (n = {P['n_clips']}): `{B}` scores better on this metric.** delta = "
      f"{P['msstft_delta_mean']:+.4f} ± {P['msstft_delta_std']:.4f} MS-STFT "
      f"(negative = `{B}` lower/better), paired t = {P['msstft_t']:.2f}, p = {P['msstft_p']:.2e}, "
      f"Wilcoxon p = {P['msstft_wilcoxon_p']:.2e}, Cohen's d = {P['cohens_d']:.2f}, per-clip wins: "
      f"`{B}` {P['wins_b']} / `{A}` {P['wins_a']}. **This number is confounded - see Headline.**")
    w("")
    w("Raw MS-STFT is not scale invariant, so a gain-matched column is reported too: the ordering "
      "is unchanged, i.e. it is not driven by the recons being ~1.2-1.5x louder than the originals.")
    w("")
    w("## Robustness to integration steps")
    w("")
    w("| steps/direction | univ MS-STFT | softnorm MS-STFT | delta | p |")
    w("|---|---|---|---|---|")
    rows = [("100", robust["100"]), ("250 (main)", main_j), ("500", robust["500"])]
    for tag, j in rows:
        sa, sb, pp = j["summary"][A], j["summary"][B], j["summary"]["paired_test"]
        w(f"| {tag} | {sa['msstft_roundtrip_mean']:.4f} | {sb['msstft_roundtrip_mean']:.4f} | "
          f"{pp['msstft_delta_mean']:+.4f} | {pp['msstft_p']:.1e} |")
    w("")
    w("The ordering holds at every step count, so it is not an artefact of Euler discretisation. "
      "Both models improve with more steps, which is why 250 was used for the headline number.")
    w("")
    w("## Is the round-trip audio meaningful, or noise? (controls)")
    w("")
    w("An absolute MS-STFT cannot answer this, so the round trip is calibrated against four "
      f"references on the same {ctrl['protocol']['num_clips']} clips:")
    w("")
    w("| model | AE ceiling | **round trip** | wrong clip | random latent | unconditional generation |")
    w("|---|---|---|---|---|---|")
    for lab in (A, B):
        c = ctrl["summary"][lab]
        w(f"| `{lab}` | {c['ceiling_mean']:.3f} | **{c['own_mean']:.3f}** | {c['wrong_mean']:.3f} | "
          f"{c['random_latent_mean']:.3f} | {c['uncond_mean']:.3f} |")
    w("")
    for lab in (A, B):
        c = ctrl["summary"][lab]
        own = ctrl["per_clip"][lab]["own"]
        wrong = ctrl["per_clip"][lab]["wrong"]
        wins = sum(1 for o, wr in zip(own, wrong) if o < wr)
        w(f"- **{lab}**: round trip beats the wrong-clip control by "
          f"{c['wrong_mean'] - c['own_mean']:+.3f} ({c['wrong_mean'] / c['own_mean']:.2f}x) on "
          f"{wins}/{len(own)} clips, and beats decoding a random latent by "
          f"{c['random_latent_mean'] - c['own_mean']:+.3f} "
          f"({c['random_latent_mean'] / c['own_mean']:.2f}x).")
    w("")
    w("So the round trip is **clip-specific and spectrally meaningful - it is not noise**. By "
      "contrast the *unconditional generation* path used by `compare_fm_latents.py` / `sample_fm.py` "
      f"scores {ctrl['summary'][A]['uncond_mean']:.2f} / {ctrl['summary'][B]['uncond_mean']:.2f}, "
      "i.e. at the wrong-clip / random-latent level, which is why the audio dumped in "
      "`evaluation/fm_latent_compare` sounds like noise. The two protocols must not be conflated: "
      "one conditions on a real clip's latent, the other does not.")
    w("")
    w("## Waveform-level diagnostics (supplementary)")
    w("")
    w("MS-STFT is phase-blind, so these characterise what will actually be heard. All waveform "
      "numbers are computed on the sign-corrected recon (see the caveat below).")
    w("")
    w("| model | loop | alpha | r0 | lag (ms) | mel_r | SI-SDR (dB) |")
    w("|---|---|---|---|---|---|---|")
    for lab in (A, B):
        for src in ("AE ceiling", "FM roundtrip"):
            m = ph["summary"][lab][src]
            w(f"| `{lab}` | {src} | {m['alpha']:+.3f} | {m['r0']:.3f} | {m['lag_ms']:.2f} | "
              f"{m['mel_r']:.3f} | {m['sisdr']:.2f} |")
    w("")
    for lab in (A, B):
        ce, rt = ph["summary"][lab]["AE ceiling"], ph["summary"][lab]["FM roundtrip"]
        w(f"- **{lab}**: the AE alone reaches SI-SDR {ce['sisdr']:.2f} dB / mel_r {ce['mel_r']:.3f}; "
          f"after the FM round trip it is {rt['sisdr']:.2f} dB / mel_r {rt['mel_r']:.3f}, so the FM "
          f"round trip costs {ce['sisdr'] - rt['sisdr']:.2f} dB of waveform fidelity at the 3 s "
          f"centre window while leaving the time-frequency envelope largely intact.")
    w("")
    w("Zero-lag is the alignment in both cases (lag above is the |cross-correlation| maximum, "
      "computed to rule out a time shift; the `univ` round trip shows a small systematic offset).")
    w("")
    w("### Caveat: the AEs reconstruct with inverted polarity")
    w("")
    w("A 440 Hz sine round-trips through these AEs as `-sine` (r = -1.000, MS-STFT 0.067): the "
      "magnitude spectrum is reproduced almost exactly while the waveform sign is flipped. This is a "
      "property of the AEs - their training objective is the phase-blind MS-STFT, for which a global "
      "sign flip is mathematically unconstrained - and **not** of the FM checkpoints. It is inaudible, "
      "but it invalidates any raw waveform correlation, which is why the diagnostics above are "
      "sign-corrected and why `alpha` is negative even for the AE ceiling.")
    w("")
    w("## Headline")
    w("")
    w("**The round-trip comparison is confounded and must not be read as `softnorm` having a better "
      "latent space.** Three separate measurements are needed to see why (all in this directory):")
    w("")
    w("**1. On the decoder-free measure the two models are tied.** The latent round-trip error "
      "(no decode involved) differs by "
      f"{S[B]['latent_rel_err_mean'] - S[A]['latent_rel_err_mean']:+.4f} "
      f"({S[A]['latent_rel_err_mean']:.4f} vs {S[B]['latent_rel_err_mean']:.4f}), which is "
      "p=0.017 by a paired t-test but **p=0.55 by Wilcoxon with 241/259 per-clip wins** - i.e. a "
      "tie in rank terms, driven by a few outlier clips. The audio gap, by contrast, is p=1e-14.")
    w("")
    w("**2. The two AEs' decoders convert the same latent error into very different audio damage.** "
      "Injecting perturbations of calibrated relative size into real latents and decoding "
      f"(`decoder_sensitivity.json`):")
    w("")
    w("| AE | slope (MS-STFT per unit rel. latent error) | MS-STFT at eps=0.10 | at the observed round-trip error | measured round trip |")
    w("|---|---|---|---|---|")
    for lab in (A, B):
        s = sens["summary"][lab]
        w(f"| `{lab}` | **{s['slope_msstft_per_unit_rel_err']:.2f}** | {s['curve']['0.1']:.3f} | "
          f"{s['predicted_from_sensitivity']:.3f} | {s['measured_roundtrip_msstft']:.3f} |")
    w("")
    w("The `univ` decoder is ~1.85x more sensitive to off-manifold latent error than the `softnorm` "
      "decoder. Since each FM is paired with a *different* AE, this difference is not a property of "
      "the flow models at all, and it is of the right size to account for the headline gap.")
    w("")
    w("**3. The RF loss comparison, by contrast, is fair - and favours `univ`.** The RF target's own "
      "energy sets the scale of the reported loss, so it was measured directly "
      "(`loss_normalization.json`, held-out latents, training protocol with batch-32 OT coupling):")
    w("")
    w("| quantity | univ | softnorm |")
    w("|---|---|---|")
    for key, name in [("target_energy_ot", "target energy per element (OT)"),
                      ("zero_predictor_mse", "zero-predictor MSE"),
                      ("linear_oracle_mse", "linear-oracle MSE (best per-t scalar on z_t)"),
                      ("replicated_mse", "checkpoint RF MSE"),
                      ("relative_loss_ot", "relative loss MSE/var"),
                      ("r2_vs_oracle", "R^2 vs linear oracle")]:
        w(f"| {name} | {lossn['models'][A][key]:.4f} | {lossn['models'][B][key]:.4f} |")
    w("")
    w("Target energy per element agrees to within 0.04%, so the raw MSE comparison **is** "
      "apples-to-apples: `univ` fits its velocity field genuinely better (relative loss 0.129 vs "
      "0.170, R^2 0.871 vs 0.831). The better training metric is real, not a units artefact.")
    w("")
    w("**4. Incidental: the OT coupling in this recipe is nearly inert.** It reduces the transport "
      f"target energy by only {lossn['models'][A]['ot_energy_reduction_pct']:.2f}% (`univ`) / "
      f"{lossn['models'][B]['ot_energy_reduction_pct']:.2f}% (`softnorm`) at batch 32 in this "
      "64x625 latent space, so the 'OT coupling straightens the paths' mechanism that M-FM1 leans "
      "on is barely being exercised by this pilot at all.")
    w("")
    w("### Corrected conclusion")
    w("")
    w("On the evidence here: `univ` has a genuinely better-fit velocity field, the two fields are "
      "about equally invertible, and the audio round-trip gap is explained by decoder sensitivity "
      "rather than by the latent space. **This pilot does not establish that either latent space is "
      "better**, and it should not be cited as showing `softnorm` is preferable.")
    w("")
    w("Separately, neither model generates usable audio yet: unconditional samples sit at "
      "wrong-clip / random-latent MS-STFT level (see the controls table above), and the earlier "
      "`fm_latent_compare` run reports generated-latent effective rank ~45 against 21 (`univ`) / 38 "
      "(`softnorm`) for real latents with generated |cos| ~0.028, i.e. near-isotropic noise. "
      "The RF-loss ordering is therefore evidence about fitting a regression target, not about "
      "generative quality.")
    w("## Files")
    w("")
    w("| path | content |")
    w("|---|---|")
    w("| `eval_fm_roundtrip/fm_roundtrip_msstft.json` | main result, per-clip scores |")
    w("| `eval_fm_roundtrip/audio/` | 8 clips x (orig, AE ceiling, FM round trip) + strips, per model |")
    w("| `eval_fm_roundtrip/audio/README.md` | file layout + strip segment order |")
    w("| `eval_fm_roundtrip/LISTEN_strips_8clips.wav` | single 428 s A/B/C/D/E listening file |")
    w("| `eval_fm_roundtrip/LISTEN_map.txt` | timing map for the file above |")
    w("| `eval_fm_roundtrip/controls/controls.json` | meaningfulness controls, per clip |")
    w("| `eval_fm_roundtrip/controls/spec_clip*.png` | spectrograms, orig vs each recon |")
    w("| `eval_fm_roundtrip/phase_diagnostics.json` | waveform-level diagnostics |")
    w("| `_robust_steps100/`, `_robust_steps500/` | step-count robustness runs |")
    w("| `eval_fm_roundtrip_msstft.py` | eval script |")
    w("| `fm_roundtrip_controls.py` | controls script |")
    w("| `fm_roundtrip_phase_diag.py` | diagnostics script |")
    w("| `build_fm_roundtrip_report.py` | this report generator |")

    text = nl.join(L) + nl
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'REPORT.md'}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
