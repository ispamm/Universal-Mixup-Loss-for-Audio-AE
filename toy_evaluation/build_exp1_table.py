#!/usr/bin/env python3
"""
Build the Exp.1 LaTeX table (single table, three sub-blocks) + metric formulas.

All values are reported as mean +- std:
  * encoder / decoder / norm-error families: 1000 validation pairs -- the SAME
    pair set the eval battery uses (generate_pairs(stems, 1000, seed=42));
  * convexity / effective rank: 256-clip pool, the SAME pool the battery uses
    (recon indices are drawn before the topo indices from one RNG stream),
    with a bootstrap std (B=500) because these are pool-level statistics.

Sources:
  toy_evaluation/eval_exp1_metrics/exp1_enc_dec_metrics.json
  toy_evaluation/eval_exp1_metrics/exp1_latent_bootstrap.json
  toy_evaluation/eval_exp1_metrics/exp1_paired_tests.json      (significance)
  toy_evaluation/eval_iclr_exp1/eval_all_summary.json           (recon MS-STFT)

Usage:  python toy_evaluation/build_exp1_table.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "toy_evaluation/eval_exp1_metrics"
# Single-audio MS-STFT decoder reconstruction quality (general fidelity, not the
# additivity/commutation MSE): the SAMELoss magnitude component summed over
# the 7 training STFT resolutions, measured as MSE(D(E(x)), x) over single clips.
# n=2000 clips, seed 42 (the reconstruction column, larger than the 1000-pair
# additivity set because per-clip recon variance is much larger). Same eval
# battery as the additivity metrics.
RECON = ROOT / "toy_evaluation/eval_iclr_exp1/eval_all_summary.json"
OUT = ROOT / "toy_evaluation/eval_iclr_exp1/exp1_table.tex"

ORDER = [
    ("transformer_minimal_full_best", "Minimal"),
    ("transformer_softMax_full_best", "Soft-Norm"),
    ("transformer_best_full_best", "Trig mixup"),
    ("transformer_implicit_full_v2_best", "Implicit (Torres)"),
    ("transformer_univ_full_best", "Universal"),
    ("transformer_univ_warm_v2_best", r"Universal (warm)$^\dagger$"),
    ("transformer_mixup_enc_best", r"\quad + encoder cone"),
    ("transformer_univ_enc_best", r"\quad Universal+enc"),
    ("univ_cycle_enc_best", r"\quad Universal+cycle"),
    ("univ_enc_anneal_kl2_best", r"\quad Univ+enc annealed"),
]

TEMPLATE = r"""% =============================================================================
%  Experiment 1 - additivity metric suite (encoder / decoder / latent space)
%  Encoder/decoder/norm-error: 1000 validation pairs (battery-identical pair set)
%  Convexity/effective rank:   256-clip pool (battery-identical) + bootstrap std
%  Reconstruction (MS-STFT):    2000 held-out clips (seed 42), same eval battery
%  2 s clips, 16 kHz, seed 42.  All entries are mean +- std.
%  Regenerate:  python toy_evaluation/build_exp1_table.py
% =============================================================================

\subsection{Additivity metrics: encoder, decoder, and latent space}
\label{sec:exp1-table}

Table~\ref{tab:exp1} reports the three metric families for all ten
configurations. Every encoder and decoder entry is a mean $\pm$ std over the
\emph{same} 1000 validation pairs used elsewhere in this paper (2\,s clips,
16\,kHz, seed 42), the latent entries use the battery's 256-clip pool with
a bootstrap standard error over clips and mix-up pairs ($B{=}500$), and the
reconstruction entry is a mean $\pm$ standard error over 2000 held-out clips
(seed 42). Lower is better except effective rank. Bold marks the best entry in
each column.
$^\dagger$Reference model used in all downstream experiments; all
$\Delta$ differences quoted in the text are paired over the identical pairs and
are significant at $p<10^{-4}$ unless stated otherwise
(see Table~\ref{tab:exp1-sig}).

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.4pt}
\caption{Exp.~1 additivity suite (mean $\pm$ std). \textbf{Encoder}: does the
encoder commute with gain-scaling (\emph{single}) and with linear interpolation
(\emph{lerp}), and how consistent is that commutation along the interpolation
path (\emph{smooth}, lower = smoother)? \textbf{Decoder}: the identical three
questions for the decoder, in units of $10^{-4}$. \textbf{Latent space}:
constant-power preservation (NormErr), manifold convexity, and effective rank.
\textbf{Recon}: single-audio MS-STFT decoder reconstruction MSE
($\mathrm{MSE}(D_\phi(E_\psi(x)),\,x)$, the training spectral loss summed over
the 7 STFT resolutions), over 2000 clips --- general reconstruction quality, not
the decoder additivity/commutation MSE of the \textbf{Decoder} block. Its
$\pm$ is the standard error of the mean ($\sigma/\sqrt{n}$, $n{=}2000$), not the
per-clip std, because the per-clip loss is heavily right-skewed. Formulas
in Eqs.~\eqref{eq:enc-single}--\eqref{eq:latent-metrics} and
Eq.~\eqref{eq:recon-msstft}.}
\label{tab:exp1}
\begin{tabular}{lcccccccccc}
\toprule
& \multicolumn{3}{c}{Encoder} & \multicolumn{3}{c}{Decoder
($\times10^{-4}$)} & \multicolumn{3}{c}{Latent space} & \multicolumn{1}{c}{Recon} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-11}
Config & single$\downarrow$ & lerp$\downarrow$ & smooth$\downarrow$
       & single$\downarrow$ & lerp$\downarrow$ & smooth$\downarrow$
       & NormErr$\downarrow$ & Conv$\downarrow$ & Eff-Rank$\uparrow$
       & MS-STFT$\downarrow$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.4pt}
\caption{Paired significance of the decoder interpolation metric
$\mathcal{E}^{\mathrm{dec}}_{\mathrm{lerp}}(0.5)$ against the reference
(Universal warm), over the identical 1000 pairs. $\Delta$ is the mean paired
difference in units of $10^{-4}$ (configuration $-$ reference), $d_z = t/\sqrt{n}$
is the paired effect size (positive $d_z$ means the configuration is
\emph{better}, i.e.\ lower, than the reference), and $n_{80}$ is the sample size
needed for $80\%$ power at $\alpha{=}0.05$. The mixup-trained configurations
differ from each other by $\sim\!10\%$ on this metric, so their ordering is
statistically resolvable but \emph{not} practically meaningful, whereas the
non-mixup configurations (Minimal, Soft-Norm) differ by $1$--$2$ orders of
magnitude.}>
\label{tab:exp1-sig}
\begin{tabular}{lrrrr}
\toprule
Config & $\Delta$ (mean) & $d_z$ & $p$ (paired $t$) & $n_{80}$ \\
\midrule
@@SIG@@
\end{tabular}
\end{table}

\paragraph{Encoder metrics.}
With $E_\psi$ the encoder, $z_i = E_\psi(x_i)$, $\alpha \in [0,1]$,
$\bar\alpha = 1-\alpha$, and the gain grid
$\mathcal{A} = \{0.25,\,0.5,\,1,\,1.5,\,2,\,3\}$:
\begin{align}
\mathcal{E}^{\mathrm{enc}}_{\mathrm{single}}
  &= \mathbb{E}_{x}\ \mathbb{E}_{a\in\mathcal{A}}
     \Bigl[\mathrm{MSE}\bigl(E_\psi(a\,x),\; a\,E_\psi(x)\bigr)\Bigr],
  \label{eq:enc-single}\\[2pt]
\mathcal{E}^{\mathrm{enc}}_{\mathrm{lerp}}(\alpha)
  &= \mathbb{E}_{x_A,x_B}
     \Bigl[\mathrm{MSE}\bigl(E_\psi(\bar\alpha x_A + \alpha x_B),\;
       \bar\alpha\,E_\psi(x_A) + \alpha\,E_\psi(x_B)\bigr)\Bigr],
  \label{eq:enc-lerp}\\[2pt]
\mathcal{S}^{\mathrm{enc}}_{\mathrm{lerp}}
  &= \mathbb{E}_{x_A,x_B}\,
     \mathrm{Var}_{\alpha\in\{0,\frac{1}{10},\dots,1\}}
     \Bigl[\mathrm{MSE}\bigl(E_\psi(\bar\alpha x_A + \alpha x_B),\;
       \bar\alpha\,E_\psi(x_A) + \alpha\,E_\psi(x_B)\bigr)\Bigr].
  \label{eq:enc-smooth}
\end{align}
The tabulated \emph{lerp} column is $\mathcal{E}^{\mathrm{enc}}_{\mathrm{lerp}}(0.5)$.
The \emph{single} metric is the homogeneity (gain) test --- the $\theta{=}0$ end
of the cone. \emph{smooth} is defined per pair and then averaged, so it carries
a standard deviation; it measures whether the homomorphism holds
\emph{uniformly} along the path rather than only at the midpoint.

\paragraph{Decoder metrics.}
Identically, with $D_\phi$ the decoder and $z_i = E_\psi(x_i)$:
\begin{align}
\mathcal{E}^{\mathrm{dec}}_{\mathrm{single}}
  &= \mathbb{E}_{x}\ \mathbb{E}_{a\in\mathcal{A}}
     \Bigl[\mathrm{MSE}\bigl(D_\phi(a\,z),\; a\,D_\phi(z)\bigr)\Bigr],
  \label{eq:dec-single}\\[2pt]
\mathcal{E}^{\mathrm{dec}}_{\mathrm{lerp}}(\alpha)
  &= \mathbb{E}_{x_A,x_B}
     \Bigl[\mathrm{MSE}\bigl(D_\phi(\bar\alpha z_A + \alpha z_B),\;
       \bar\alpha\,D_\phi(z_A) + \alpha\,D_\phi(z_B)\bigr)\Bigr],
  \label{eq:dec-lerp}\\[2pt]
\mathcal{S}^{\mathrm{dec}}_{\mathrm{lerp}}
  &= \mathbb{E}_{x_A,x_B}\,
     \mathrm{Var}_{\alpha\in\{0,\frac{1}{10},\dots,1\}}
     \Bigl[\mathrm{MSE}\bigl(D_\phi(\bar\alpha z_A + \alpha z_B),\;
       \bar\alpha\,D_\phi(z_A) + \alpha\,D_\phi(z_B)\bigr)\Bigr].
  \label{eq:dec-smooth}
\end{align}
Both $\mathcal{E}^{\mathrm{dec}}_{\mathrm{lerp}}(\alpha)$ and
$\mathcal{E}^{\mathrm{enc}}_{\mathrm{lerp}}(\alpha)$ vanish iff the module
commutes \emph{exactly} with linear interpolation; the decoder metric is
therefore the waveform-domain statement of decoder additivity, and the encoder
metric is its latent-domain counterpart.

\paragraph{Latent-space metrics.}
For a pool of $N$ codes $z_i\in\mathbb{R}^P$, a pair $(z_A,z_B)$, and the
constant-power weights $c=\cos\tfrac{\pi}{4}=s=\sin\tfrac{\pi}{4}$ at
$\alpha{=}0.5$:
\begin{align}
\mathcal{N}
  &= \mathbb{E}_{z_A,z_B}
    \Bigl[\bigl|\,\|c\,z_A + s\,z_B\| - \|z_A\|\,\bigr|\Bigr],
  \label{eq:norm-err}\\[2pt]
\mathcal{C}
  &= \mathbb{E}_{z_A,z_B}
    \Bigl[\min_{z\in\mathcal{Z}}
      \bigl\|\,(c\,z_A + s\,z_B) - z\,\bigr\|\Bigr],
  \label{eq:conv}\\[2pt]
\mathcal{R}
  &= \frac{\bigl(\sum_{i}\sigma_i^2\bigr)^2}{\sum_{i}\sigma_i^4},
  \label{eq:latent-metrics}
\end{align}
where $\mathcal{Z}=\{z_i\}_{i=1}^N$ is the latent pool and $\sigma_i$ are the
singular values of the centred pooled-code matrix. $\mathcal{N}$ is $0$ iff the
latent mixture preserves the constant-power norm exactly; $\mathcal{C}$ is the
nearest-neighbour distance of the mixture midpoint to the pool (endpoints
excluded), so a low value means the manifold is ``filled in'' between its
samples; $\mathcal{R}$ is the participation-ratio effective rank
($\mathcal{R}\le P$, with $\mathcal{R}=P$ iff the codes are isotropic). Both
$\mathcal{R}$ and $\mathcal{C}$ are pool statistics, so their std is a
bootstrap standard error, not a per-sample std.

\paragraph{Reconstruction metric.}
The \emph{Recon} column reports general single-audio decoder reconstruction
quality --- distinct from the \emph{Decoder} block above, which measures
additivity/commutation, not fidelity. With $D_\phi$ the decoder and $E_\psi$
the encoder:
\begin{equation}
\mathcal{L}_{\mathrm{MS\text{-}STFT}}
  = \mathbb{E}_{x}\ \mathcal{L}^{\mathrm{mag}}_{\mathrm{STFT}}
    \bigl(D_\phi(E_\psi(x)),\; x\bigr),
  \label{eq:recon-msstft}
\end{equation}
where $\mathcal{L}^{\mathrm{mag}}_{\mathrm{STFT}}$ is the \texttt{SAMELoss}
magnitude objective the models were trained with (spectral contrast plus
adaptive log-magnitude, summed over the 7 STFT resolutions
$\mathrm{FFT}\in\{32,\dots,2048\}$, 75\,\% overlap, Hann window). Lower is
better. It is evaluated on 2000 held-out clips (seed 42) --- more than the
1000-pair additivity set because per-clip reconstruction variance is much
larger, as noted in the sample-size discussion below. The reported $\pm$ is the
standard error of the mean $\sigma/\sqrt{n}$ ($n{=}2000$), \emph{not} the
per-clip standard deviation: the per-clip loss is heavily right-skewed (median
$\approx 0.3$, mean $\approx 0.5$, tail to $\approx 3.5$), so the per-clip std
($\approx 0.6$, larger than the mean) is uninformative as an uncertainty on the
mean, whereas the SEM is $\approx 0.013$ ($\sim\!2\%$ of the mean).

\paragraph{What the suite shows.}
\begin{enumerate}
\item \textbf{Cone supervision is necessary for decoder commutation.}
      Without a mixup loss the decoder does not commute with mixing
      ($\mathcal{E}^{\mathrm{dec}}_{\mathrm{lerp}}$: Minimal
      $17.8\times10^{-4}$) and the soft-norm-only model is worse still
      ($307\times10^{-4}$): the bottleneck normalisation destroys the magnitude
      algebra that mixing relies on. Every cone-trained model drops to
      $0.9$--$1.3\times10^{-4}$, a $15$--$340\times$ reduction.
\item \textbf{The encoder side follows the same ordering, but with a cost.}
      The encoder-aligned configurations dominate
      $\mathcal{E}^{\mathrm{enc}}_{\mathrm{lerp}}$ (Universal+cycle
      $0.0042\pm0.0067$, Universal+enc $0.0195\pm0.0331$, versus
      $0.52\pm0.06$--$0.80\pm0.17$ for the decoder-only models), and they are
      also the smoothest along the path
      ($\mathcal{S}^{\mathrm{enc}}_{\mathrm{lerp}}\!\sim\!10^{-6}$--$10^{-5}$
      against $\sim\!10^{-2}$). The price is visible in NormErr: every
      encoder-aligned model sits at $\mathcal{N}\approx30$--$32$, an order of
      magnitude above the $3.1$--$3.4$ of the decoder-only models.
\item \textbf{Encoder alignment and latent regularity are antagonistic.}
      Sorting by $\mathcal{N}$ gives annealed ($3.12\pm2.73$) $\approx$
      warm ($3.40\pm3.14$) $<$ Trig/Implicit ($\sim3.1$) $\ll$
      encoder-aligned ($\sim30$), while sorting by encoder homomorphism gives
      almost the reverse order. The annealed run was designed to escape this
      trade-off and it does recover $\mathcal{N}$ and a high rank
      ($82.6\pm2.7$), but it gives back the encoder alignment
      ($\mathcal{E}^{\mathrm{enc}}_{\mathrm{lerp}}=0.486$) and leaves the
      decoder damaged at low gain (next point).
\item \textbf{The gain test and the mix test measure different things, and
      they disagree for the annealed model.}
      Univ+enc annealed is among the \emph{best} on decoder interpolation
      ($1.31\times10^{-4}$) and by far the \emph{worst} on decoder gain
      ($7391\times10^{-4}$, $\sim\!1500\times$ worse than Trig mixup). The
      cause is a genuine decoder pathology that we verified is not a
      checkpoint-loading artifact (0 missing/unexpected keys, no non-finite
      tensors): for this checkpoint $D_\phi(0)$ is not silence but noise
      (rms $0.43$, peak $1.80$; at $a{=}0.25$ the output peaks at $6.9$),
      whereas the reference model gives $D_\phi(0)=0$ exactly and scales
      monotonically. A report quoting only interpolation metrics would miss
      this entirely --- hence the separate \emph{single} column. It also
      matters in practice: the training recipe snaps gains $|a|<0.05$ to $0$
      to synthesise silence, which presupposes $D_\phi(0)=0$.
\item \textbf{Sample size.} At $n=1000$ pairs the standard error of
      $\mathcal{E}^{\mathrm{dec}}_{\mathrm{lerp}}$ is
      $5$--$9\times10^{-6}$, i.e.\ $\sim\!5\%$ of the values being compared,
      and it falls as $n^{-1/2}$; the paired tests in Table~\ref{tab:exp1-sig}
      are all significant, but the effect sizes show that the
      \emph{differences among the cone-trained models} are small enough that
      only the large gaps (mixup vs no-mixup, encoder-aligned vs not) should be
      read as substantive. Reconstruction metrics need more samples than these
      latent-geometry metrics because their per-clip variance is much larger,
      which is why the reconstruction column of our tables uses 2000 clips.
\end{enumerate}
"""


def sci(v, d=1):
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    m, e = f"{v:.{d}e}".split("e")
    return "$" + m + r"\!\times\!10^{" + str(int(e)) + "}$"


def ms(mean, std, fmt_mean, fmt_std=None):
    """format 'mean +- std' with a per-column mean formatter."""
    if fmt_std is None:
        fmt_std = fmt_mean
    return "$" + fmt_mean(mean) + r" \pm " + fmt_std(std) + "$"


def build():
    m = json.loads((M / "exp1_enc_dec_metrics.json").read_text())
    b = json.loads((M / "exp1_latent_bootstrap.json").read_text())
    sig_path = M / "exp1_paired_tests.json"
    sig = json.loads(sig_path.read_text()) if sig_path.exists() else {}
    # single-audio MS-STFT decoder reconstruction quality (general fidelity)
    recon = json.loads(RECON.read_text())["checkpoints"] if RECON.exists() else {}

    rows = []
    for key, disp in ORDER:
        if key not in m:
            print(f"  [warn] {key} missing")
            continue
        r, bb = m[key], b.get(key, {})
        rr = recon.get(key, {}).get("protocols", {}).get("reconstruction", {})
        rows.append({
            "key": key, "display": disp,
            "enc_single": (r["enc_single"]["mean"], r["enc_single"]["std"]),
            "enc_lerp": (r["enc_lerp@0.5"]["mean"], r["enc_lerp@0.5"]["std"]),
            "enc_smooth": (r["enc_lerp_smoothness"]["mean"], r["enc_lerp_smoothness"]["std"]),
            "dec_single": (r["dec_single"]["mean"], r["dec_single"]["std"]),
            "dec_lerp": (r["dec_lerp@0.5"]["mean"], r["dec_lerp@0.5"]["std"]),
            "dec_smooth": (r["dec_lerp_smoothness"]["mean"], r["dec_lerp_smoothness"]["std"]),
            "norm": (r["norm_error"]["mean"], r["norm_error"]["std"]),
            "conv": (bb.get("convexity_trig"), bb.get("convexity_boot_std")),
            "rank": (bb.get("effective_rank"), bb.get("effective_rank_boot_std")),
            "recon": (rr.get("mean_msstft"), rr.get("std_msstft")),
            "recon_n": rr.get("num_clips_valid"),
        })

    # best = smallest mean, except rank (largest)
    cols = ["enc_single", "enc_lerp", "enc_smooth", "dec_single", "dec_lerp",
            "dec_smooth", "norm", "conv", "rank", "recon"]
    tops = {}
    for c in cols:
        vals = [r[c][0] for r in rows if r[c][0] is not None and np.isfinite(r[c][0])]
        tops[c] = max(vals) if c == "rank" else min(vals)

    def cell(r, c):
        mean, std = r[c]
        if mean is None or not np.isfinite(mean):
            return "--"
        if c in ("enc_smooth", "dec_smooth"):
            txt = "$" + sci(mean) + r" \pm " + sci(std) + "$"
        elif c == "dec_single":
            txt = ms(mean * 1e4, std * 1e4, lambda v: f"{v:.2f}")
        elif c == "dec_lerp":
            txt = ms(mean * 1e4, std * 1e4, lambda v: f"{v:.2f}")
        elif c == "recon":
            # report the standard ERROR of the mean (std/sqrt(n)), not the
            # per-clip std: the MS-STFT loss is heavily right-skewed (median
            # ~0.3, mean ~0.5, tail to ~3.5), so the per-clip std (~0.6, larger
            # than the mean) is uninformative as a "+/-" on the mean. With
            # n=2000 the SEM is ~0.013 (~2% of the mean).
            n = r.get("recon_n", 0) or 1
            sem = std / np.sqrt(n)
            txt = ms(mean, sem, lambda v: f"{v:.3f}")
        else:
            prec = 3 if c == "enc_single" else 4 if c == "enc_lerp" else 2
            txt = ms(mean, std, lambda v: f"{v:.{prec}f}")
        return r"\textbf{" + txt + "}" if abs(mean - tops[c]) < 1e-12 else txt

    body = []
    for r in rows:
        body.append(" & ".join([r["display"]] + [cell(r, c) for c in cols]) + r" \\")

    # significance table body
    n_pairs = None
    for r in rows:
        if r["key"] in m:
            n_pairs = m[r["key"]]["dec_lerp@0.5"]["n"]
            break
    sigbody = []
    for r in rows:
        k = r["key"]
        if k not in sig or "dec_lerp@0.5" not in sig[k]:
            continue
        d = sig[k]["dec_lerp@0.5"]
        # exact paired effect size: t = dbar / (s_d/sqrt(n))  =>  d_z = t / sqrt(n)
        dz = d["t"] / np.sqrt(n_pairs)
        n80 = int(np.ceil((2.8 / (abs(dz) + 1e-30)) ** 2))
        sigbody.append(f"{r['display']} & ${d['delta']*1e4:+.3f}$ & ${dz:+.3f}$ & "
                       f"${d['p']:.1e}$ & {n80} \\\\")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("@@BODY@@", "\n".join(body))
                   .replace("@@SIG@@", "\n".join(sigbody)))
    print(f"Wrote {OUT}\n")

    hdr = f"{'config':<24}{'encS':>17}{'encL':>17}{'decL(x1e-4)':>17}{'norm':>16}{'rank':>14}"
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        nm = r["display"].replace(r"\quad ", "  ").replace(r"$^\dagger$", "*")
        print(f"{nm:<24}"
              f"{r['enc_single'][0]:>9.3f}+-{r['enc_single'][1]:<6.3f}"
              f"{r['enc_lerp'][0]:>9.4f}+-{r['enc_lerp'][1]:<6.4f}"
              f"{r['dec_lerp'][0]*1e4:>9.3f}+-{r['dec_lerp'][1]*1e4:<6.3f}"
              f"{r['norm'][0]:>8.2f}+-{r['norm'][1]:<6.2f}"
              f"{r['rank'][0]:>8.2f}+-{r['rank'][1]:<5.2f}")


if __name__ == "__main__":
    build()
