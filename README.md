# Universal Mixup Loss for Audio Autoencoders

Training and evaluation code for **universal mixup losses** that make the latent
space of an audio autoencoder *additive* — i.e. trained so that

```
encode(A + B) ≈ encode(A) + encode(B)
decode(α·Z_A + (1−α)·Z_B) ≈ α·decode(Z_A) + (1−α)·decode(Z_B)
```

holds for arbitrary mixtures of signals, without collapsing the latent space.

The object of the study is a trainable "toy" Transformer audio autoencoder
(16 kHz, 2 s chunks → `(B, 125, 64)` latents, 4:1 compression). Training
combines — all switchable via CLI — reconstruction, a multi-resolution STFT loss,
an adversarial STFT PatchGAN, phase, norm-preservation and the
**universal (trigonometric) mixup / "cone"** additivity loss, and the
evaluation suite measures how additive / topologically linear the learned
latent space actually is (decoder additivity LIMS, encoder homomorphism,
latent convexity, layer-wise linearity probes, …).

---

## Repository layout

```
toy_model/             The trainable toy autoencoders (the object of the study)
  transformer_autoencoder.py   ToyTransformerAutoencoder  ← main model
  autoencoder.py               ToyConvAutoencoder         ← CNN ablation baseline
  activations.py               SnakeActivation, ...
  discriminator.py             STFTDiscriminator (PatchGAN)
toy_training/          Standalone training loop with every loss switchable via CLI
  train_toy.py                 training loop + CLI (universal_mix / cone loss, ... )
  losses.py                    SAMELoss (multi-res STFT) + additivity losses
  train_fm_pilot.py / train_fm_cond.py / sample_fm.py
                               latent flow-matching (rectified-flow) decoders
  bench_step.py / hparam_report.py / hparam_search_driver.py (+ example queue json)
  run_seed_comparison.sh / run_align_nocollapse.sh / run_sampler_ablation.sh ...
                               reference training recipes
toy_evaluation/        Additivity / linearity measurement suite
  lims_test_toy.py            decoder additivity (Lerp/Slerp/universal LSD + energy)
  latent_distance_toy.py       encoder homomorphism (interp(Z_A,Z_B) vs encode(mix))
  latent_topology_toy.py      latent density / convexity / anisotropy
  eval_reconstruction.py      reconstruction fidelity (MS-STFT)
  evaluate_all.py             full battery over arbitrary checkpoints
  generalized_interpolation_eval.py / generalized_decoder_path_eval.py
                               non-linear interpolation-weight laws
  deep_additivity_eval.py      layer-wise mixup commutation, mix-at-any-depth
  layer_linearity_eval.py / layer_spectrogram_eval.py / attention_map_eval.py
                               layer-by-layer linearity probes
  unified_ablation/           unified loss ablation (six configs)
  separation_compare/         algebraic source separation on MUSDB18-HQ chunks
  data.py / model_utils.py    shared data loading + checkpoint wrappers
```

## Installation

Requires Python ≥ 3.10 and CUDA 12.6 wheels for PyTorch 2.7.1 (CPU-only also
works, slowly). The recommended way is [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # creates .venv from uv.lock (exact pinned environment)
source .venv/bin/activate
```

or with plain pip:

```bash
pip install -e .
```

`wandb` (experiment tracking, optional) and `matplotlib` (some eval plots) are
used by a few scripts — install them if needed.

## Data

The training/eval scripts consume any directory tree of `.wav` files, passed
via `--data_dir`. CLI defaults point to placeholder paths under `/data/...`
(edit them or pass your own). In our experiments we used openly available
corpora of speech, environmental sounds and music:

- LibriTTS `wav48_silence_trimmed` (16 kHz speech chunks)
- FSD50K `FSD50K.dev_audio` (environmental sounds)
- FMA `fma_small` (music)
- MUSDB18-HQ (for `separation_compare/`, 44.1 kHz)

Datasets are external to this repo. A preprocessed-cache directory
(`--preprocess_cache_dir`, default `/data/preprocessed_cache`) speeds up
repeat runs.

## Training

```bash
python toy_training/train_toy.py \
    --data_dir /data/wav48_silence_trimmed \
               /data/fsd50k/FSD50K.dev_audio \
               /data/fma_small \
    --model_type transformer \
    --use_universal_mix --no_use_latent_mixup \
    --use_soft_norm --use_gan \
    --sample_rate 16000 --latent_dim 64 --patch_size 256 \
    --epochs 200 --run_name universal_mix_run
```

Reference recipes (seeds, GAN weight, scheduler, plateau patience) are in
`toy_training/run_seed_comparison.sh` and
`toy_training/run_align_nocollapse.sh`.

A hyperparameter-search driver with an example queue is available:

```bash
python toy_training/hparam_search_driver.py \
    --queue toy_training/hparam_search_queue.json --dry-run
```

## Evaluation

```bash
# full battery over one or more checkpoints (recon, additivity, topology)
python toy_evaluation/evaluate_all.py \
    --checkpoints toy_training/checkpoints/<run_name>_best.pt

# decoder-additivity (LIMS) comparison: baseline vs universal-mixup model
python toy_evaluation/lims_test_toy.py \
    --ckpt_baseline toy_training/checkpoints/<baseline_run>_best.pt \
    --ckpt_phase toy_training/checkpoints/<universal_mix_run>_best.pt

# encoder homomorphism: interp(Z_A, Z_B) vs encode(mix)
python toy_evaluation/latent_distance_toy.py \
    --ckpt_baseline toy_training/checkpoints/<baseline_run>_best.pt \
    --ckpt_phase toy_training/checkpoints/<universal_mix_run>_best.pt
```

Checkpoints are written as `toy_training/checkpoints/<run_name>_best.pt` and
are **not** versioned. Every evaluation script fixes its own pair-generation
seed, so different checkpoints are scored on identical held-out data.

## External baselines

`toy_evaluation/separation_compare/worker_same_orig*.py` compare against the
original **SAME-L** autoencoder and require the
[stable-audio-3](https://github.com/Stability-AI/stable-audio-3) package on
`sys.path`. `worker_dac*.py` / `worker_cae*.py` compare against
[DAC](https://github.com/descriptinc/descript-audio-codec) and
[Lin-CAE](https://github.com/lca0/linear_cae) respectively and need the
corresponding packages. All baseline weights are downloaded from Hugging Face
on first use (cached under `.hf_cache/`).

## License

MIT — see `LICENSE`.
