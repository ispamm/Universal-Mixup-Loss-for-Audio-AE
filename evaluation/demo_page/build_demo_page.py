#!/usr/bin/env python3
"""
Experiment 4 (rebuttal): build the static, anonymous demo page (index.html).

Reads cases.json + the per-model manifests written by generate_demo_audio.py
and emits a self-contained HTML page (no external dependencies; audio via
relative paths) that a reviewer can open directly or that can be dropped into
the anonymous repository / hosted statically.

    .venv/bin/python toy_evaluation/demo_page/build_demo_page.py
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = [
    ("univ", "Universal Mixup (Ours)", "toy transformer, 44.1 kHz, cone-supervised"),
    ("same_l", "SAME-L", "production SAME, no additivity training"),
    ("lin_cae", "Lin-CAE", "linear-consistency autoencoder (Torres et al.)"),
    ("dac", "DAC (44 kHz)", "descript audio codec, RVQ-quantized"),
]


def load():
    cases = json.loads((HERE / "cases.json").read_text())
    mans = {}
    for m, _, _ in MODELS:
        p = HERE / "audio" / f"manifest_{m}.json"
        if not p.exists():
            raise SystemExit(f"[missing] {p} -- run generate_demo_audio.py --model {m}")
        mans[m] = json.loads(p.read_text())["files"]
    return cases, mans


def value_badge(info, best):
    """SI-SDR chip (shown ONLY in Section 4, so Sections 1-3 stay unbiased)."""
    sisdr = info.get("sisdr")
    if sisdr is None:
        return '<span class="badge na">n/a</span>'
    cls = "best" if abs(sisdr - best) < 1e-9 else ("bad" if sisdr < 0 else "")
    return f'<span class="badge {cls}">{sisdr:+.2f} dB</span>'


def guard_notes(info):
    """Safe-listening caveat for genuine output-level failures (e.g. DAC's
    separation). Ordinary peak-normalisation during the float->16-bit
    conversion is NOT flagged: every decoder reproduces hot source levels
    (the corpus is peak-normalised to 0.95 and raw-sum mixtures exceed it),
    and the sub-dB trim is standard, inaudible conversion hygiene."""
    notes = ""
    if info.get("loudness_guarded"):
        notes += (f' <span class="att" title="the model output was dangerously loud; '
                  f'attenuated for safe listening (the level failure is part of the '
                  f'result)">&#9888; volume reduced &times;{info["gain"]:.2f}</span>')
    return notes


def audio_cell(rel, info, best=None, extra=None, show_value=False):
    src = f"audio/{rel}"
    b = (value_badge(info, best) if (show_value and best is not None) else "") \
        + guard_notes(info)
    ex = f'<div class="sub">{extra}</div>' if extra else ""
    return (f'<td><audio controls preload="none" src="{src}"></audio>'
            f'<div class="badges">{b}</div>{ex}</td>')


def ref_cell(rel):
    return f'<td><audio controls preload="none" src="audio/{rel}"></audio></td>'


def best_of(infos):
    vals = [i.get("sisdr") for i in infos if i.get("sisdr") is not None]
    return max(vals) if vals else None


def track_name(tid):
    return tid.split("__", 1)[1] if "__" in tid else tid


def build():
    cases, mans = load()
    S = []

    S.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Universal Mixup Loss — Audio Demonstrations</title>
<style>
  :root { --ink:#1a1a2e; --acc:#0f6cbd; --soft:#f5f6fa; --line:#d9dbe6; }
  body { font-family: Georgia, 'Times New Roman', serif; margin: 0; color: var(--ink); background:#fff; }
  header { background: linear-gradient(135deg,#101a33,#1d2a52); color:#fff; padding:42px 6vw 34px; }
  header h1 { margin:0 0 10px; font-size: 30px; font-weight:600; letter-spacing:.3px; }
  header p { margin:4px 0; max-width: 900px; color:#d6ddf3; font-size:15px; line-height:1.55; }
  header code { background:rgba(13,23,58,.85); color:#aecbff;
               border:1px solid rgba(174,203,255,.28); }
  main { padding: 28px 6vw 60px; max-width: 1280px; margin: 0 auto; }
  h2 { font-size: 21px; margin: 42px 0 6px; border-bottom: 2px solid var(--acc); padding-bottom: 6px; }
  h2 .num { color: var(--acc); margin-right: 8px; }
  p.proto { color:#444; font-size: 14.5px; line-height: 1.6; max-width: 980px; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0 8px; }
  th, td { border: 1px solid var(--line); padding: 10px 12px; vertical-align: top; font-size: 14px; }
  th { background: var(--soft); font-family: Helvetica, Arial, sans-serif; font-size: 13px; text-align:left; }
  td.rowhead { font-weight: bold; background:#fafbfd; min-width: 190px; }
  td.rowhead .sub { font-weight: normal; color: #666; font-size: 12.5px; }
  audio { width: 230px; height: 34px; display:block; }
  .sub audio { width: 210px; height: 28px; }
  .badges { margin-top: 6px; font-family: Helvetica, Arial, sans-serif; font-size: 12px; }
  .badge { display:inline-block; padding: 2px 8px; border-radius: 10px; background:#eef1f6; color:#173b63; }
  .badge.best { background:#0f6cbd; color:#fff; font-weight:600; }
  .badge.bad { background:#fdecea; color:#a4271a; }
  .badge.na { background:#eee; color:#888; }
  .sub { color:#777; font-size: 12px; margin-top:4px; font-family: Helvetica, Arial, sans-serif; }
  .att { color:#a46a00; }
  code { font-family: Menlo, Consolas, monospace; background:#eef1f6; color:#173b63;
         padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
  .disc { background: var(--soft); border-left: 4px solid var(--acc); padding: 14px 18px;
          font-size: 13.5px; line-height:1.6; color:#333; margin-top: 40px; }
  footer { color:#888; font-size: 12.5px; margin-top: 26px; }
</style>
</head>
<body>
<header>
  <h1>Universal Mixup Loss: Audio Demonstrations</h1>
  <p>Listening examples for <em>&ldquo;Universal Mixup Loss: Training Audio Autoencoders to Decode the
     Positive-Weight Mixing Cone&rdquo;</em>. Four autoencoders are compared on identical clips at
     44.1&nbsp;kHz: <b>Universal Mixup (ours)</b>, <b>SAME-L</b> (production),
     <b>Lin-CAE</b> (linear-consistency baseline), and <b>DAC</b> (44&nbsp;kHz codec).</p>
  <p>All operations are performed <b>in latent space only</b> and then decoded: reconstruction
     <code>D(E(x))</code>, interpolation <code>D(w1(&alpha;)&middot;z_A + w2(&alpha;)&middot;z_B)</code>,
     and algebraic separation <code>D(z_mix &minus; z_other)</code>. The reference in every
     experiment is the corresponding <b>audio-domain</b> operation. Sections 1&ndash;3 are presented
     <b>without scores</b>, so you can judge by ear; in Section 4 the number under each player is
     the SI-SDR of exactly the audio you hear (scale-invariant, dB). Only one player plays at a
     time &mdash; starting one stops the previous.</p>
</header>
<main>
""")

    # ---------------- 1. Reconstruction ----------------
    S.append('<h2><span class="num">1.</span>Audio reconstruction (control)</h2>')
    S.append('<p class="proto">Sanity anchor: every model encodes and decodes the clip (MUSDB18-HQ, '
             '2&nbsp;s) without any latent arithmetic. The baselines are competent codecs &mdash; '
             'which is what makes the algebraic differences below meaningful.</p>')
    S.append('<table><tr><th>Clip</th><th>Ground truth</th>' +
             "".join(f"<th>{d}</th>" for _, d, _ in MODELS) + "</tr>")
    for c in cases["reconstruction"]:
        base = f"reconstruction/{c['case']}"
        infos = [mans[m][f"{base}/{m}.wav"] for m, _, _ in MODELS]
        S.append(f"<tr><td class='rowhead'>{track_name(c['track'])}<div class='sub'>{c['stem']}</div></td>")
        S.append(ref_cell(f"{base}/ground_truth.wav"))
        for (m, _, _), info in zip(MODELS, infos):
            S.append(audio_cell(f"{base}/{m}.wav", info))
        S.append("</tr>")
    S.append("</table>")

    # ---------------- 2/3. Interpolation ----------------
    for num, law, title, formula, note in (
        ("2.", "lerp", "Linear interpolation (lerp)",
         "<code>D((1&minus;&alpha;)&middot;z_A + &alpha;&middot;z_B)</code>", ""),
        ("3.", "trig", "Trigonometric interpolation (constant-power)",
         "<code>D(cos(&alpha;&pi;/2)&middot;z_A + sin(&alpha;&pi;/2)&middot;z_B)</code>",
         " (different stem pairs than Section 2, so both laws are heard on fresh content)")):
        S.append(f'<h2><span class="num">{num}</span>{title}</h2>')
        S.append(f'<p class="proto">Latent-space crossfade {formula} compared against the '
                 f'audio-domain crossfade of the same stems{note}. Each file sweeps '
                 '&alpha; = 0 &rarr; 1 in 9 steps (2&nbsp;s per step, 0.2&nbsp;s of silence '
                 'between steps), so you hear the entire path &mdash; the mid-path behaviour '
                 'is where decoders without cone supervision break down. Clips: MUSDB18-HQ, 2&nbsp;s.</p>')
        S.append('<table><tr><th>Crossfade</th><th>Audio-domain ideal</th>' +
                 "".join(f"<th>{d}</th>" for _, d, _ in MODELS) + "</tr>")
        for c in cases["interpolation"][law]:
            base = f"interpolation/{law}/{c['case']}"
            infos = [mans[m][f"{base}/{m}.wav"] for m, _, _ in MODELS]
            S.append(f"<tr><td class='rowhead'>{track_name(c['track'])}"
                     f"<div class='sub'>{c['stem_a']} &rarr; {c['stem_b']}</div></td>")
            S.append(ref_cell(f"{base}/ideal.wav"))
            for (m, _, _), info in zip(MODELS, infos):
                S.append(audio_cell(f"{base}/{m}.wav", info))
            S.append("</tr>")
        S.append("</table>")

    # ---------------- 4. Separation (synthetic, in-distribution) ----------------
    S.append('<h2><span class="num">4.</span>Algebraic source separation (in-distribution synthetic mixtures)</h2>')
    S.append('<p class="proto">Mixtures are built <b>artificially from the autoencoders&rsquo; own '
             'training corpus</b> (VCTK voice + FSD50K/FMA interference, natural raw sums, no '
             'renormalisation): <code>mix = voice + other</code>. The voice is then extracted '
             '<b>without any separator network</b> by pure latent algebra, '
             '<code>D(z_mix &minus; z_other)</code> (oracle access to the interfering stem&rsquo;s '
             'code, as in the paper&rsquo;s in-distribution protocol). Under each model&rsquo;s '
             'separation we also provide its own <b>reconstruction ceiling</b> &mdash; '
             '<code>D(E(voice))</code> &mdash; which shows that the baselines&rsquo; failure is '
             '<em>algebraic</em>, not a codec-quality artifact: their ceilings are fine, their '
             'subtractions are not.</p>')
    S.append('<table><tr><th>Case</th><th>Mixture</th><th>Voice (ground truth)</th>' +
             "".join(f"<th>{d}</th>" for _, d, _ in MODELS) + "</tr>")
    for c in cases["separation_synth"]:
        base = f"separation/{c['case']}"
        infos = [mans[m][f"{base}/{m}.wav"] for m, _, _ in MODELS]
        best = best_of(infos)
        tag = "music (FMA)" if c["other_tag"] == "fma" else "sound event (FSD50K)"
        S.append(f"<tr><td class='rowhead'>voice + {tag}<div class='sub'>interference: "
                 f"{c['interference'][:38]}</div></td>")
        S.append(ref_cell(f"{base}/mixture.wav"))
        S.append(ref_cell(f"{base}/ground_truth.wav"))
        for (m, _, _), info in zip(MODELS, infos):
            ceil = mans[m][f"{base}/{m}_ceiling.wav"]
            extra = (f"ceiling (own recon): {ceil['sisdr']:+.2f} dB "
                     f'<audio controls preload="none" src="audio/{base}/{m}_ceiling.wav"></audio>')
            S.append(audio_cell(f"{base}/{m}.wav", info, best, extra=extra, show_value=True))
        S.append("</tr>")
    S.append("</table>")

    # ---------------- disclosure ----------------
    S.append("""
<div class="disc">
<b>Protocol &amp; selection.</b> All models run on
identical clips; every latent operation is followed by that model&rsquo;s own decoder. Displayed
cases were selected from the paper's full quantitative evaluations.  All files are peak-normalised to 0.95 when
converted to 16-bit PCM. The only flagged case is an actual output-level failure: &ldquo;&#9888; volume reduced&rdquo; marks a model whose
separation output was dangerously loud and was attenuated for safe listening &mdash; the level
failure itself is part of the result. Lin-CAE decodes stochastically; a fixed decode seed is used
for reproducibility.
</div>
<footer>Anonymous demo page for the ICLR 2027 rebuttal. Static HTML; open directly or host the
whole folder. Audio: 16-bit PCM WAV, 44.1 kHz.</footer>
<script>
(function () {
  // Only one player at a time: starting an audio stops every other one.
  var players = Array.prototype.slice.call(document.querySelectorAll('audio'));
  players.forEach(function (p) {
    p.addEventListener('play', function () {
      players.forEach(function (q) {
        if (q !== p && !q.paused) { q.pause(); }
      });
    });
  });
})();
</script>
</main></body></html>""")

    out = HERE / "index.html"
    out.write_text("\n".join(S))
    print(f"Saved: {out}")
    n_audio = len(list((HERE / "audio").rglob("*.wav")))
    size = sum(f.stat().st_size for f in (HERE / "audio").rglob("*.wav"))
    print(f"{n_audio} wav files, {size/1e6:.1f} MB total")


if __name__ == "__main__":
    build()
