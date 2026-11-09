# Universal Mixup Loss for Audio Autoencoders

Code for **"Universal Mixup Loss: Training Audio Autoencoders to Decode the
Positive-Weight Mixing Cone"** — training and evaluating audio autoencoders
whose latent space is *additive*:

```
encode(A + B) ≈ encode(A) + encode(B)
decode(w1·Z_A + w2·Z_B) ≈ w1·decode(Z_A) + w2·decode(Z_B)
```

so latent codes support mixing, crossfading, interpolation and source
separation by plain vector arithmetic — with no extra parameters or compute at
inference time.

## Repository layout

| Path | Content |
|---|---|
| `model/` | Transformer autoencoder (16/44.1 kHz, 64-channel continuous bottleneck) and the conv ablation baseline, STFT PatchGAN discriminator |
| `training/` | Training loop with every loss switchable via CLI (reconstruction, multi-res STFT, adversarial, phase, norm preservation, **universal mixup / cone loss**); rectified-flow DiT training over frozen latents |
| `evaluation/` | Additivity & linearity battery (decoder commutation, encoder homomorphism, latent topology, layer-wise probes), held-out interpolation-law suite, algebraic separation vs SAME-L / Lin-CAE / DAC |
| `evaluation/demo_page/` | **Audio demo page** — open `demo_page/index.html` in a browser |

## Quick start

Requires Python ≥ 3.10 (CUDA 12.6 wheels for PyTorch 2.7.1; CPU works, slowly).

```bash
uv sync                  # or: pip install -e .
```

Train the universal-mixup autoencoder:

```bash
python training/train.py \
    --data_dir <wav48_silence_trimmed> <FSD50K.dev_audio> <fma_small> \
    --model_type transformer --use_universal_mix --no_use_latent_mixup \
    --use_soft_norm --use_gan --sample_rate 16000 \
    --latent_dim 64 --patch_size 256 --epochs 200 --run_name universal_mix_run
```

Evaluate a checkpoint (reconstruction, additivity, topology, in one pass):

```bash
python evaluation/evaluate_all.py --checkpoints training/checkpoints/universal_mix_run_best.pt
```

Checkpoints are written to `training/checkpoints/` and not versioned. Every
evaluation script fixes its own pair-generation seed, so different
checkpoints are scored on identical held-out data.

## Audio demo

`evaluation/demo_page/index.html` is a self-contained static page comparing the
universal-mixup autoencoder against SAME-L, Lin-CAE and DAC on identical
clips: reconstruction, linear and trigonometric latent interpolation, and
algebraic source separation (`D(z_mix − z_other)`), with per-model
reconstruction ceilings. Regenerate it with the scripts in the same folder.

## Data

Scripts consume any directory tree of `.wav` files via `--data_dir`. The
paper's experiments use LibriTTS-R (`wav48_silence_trimmed`), FSD50K and FMA
`fma_small`; MUSDB18-HQ is used for the 44.1 kHz separation comparisons. All
datasets are external to this repository.

## External baselines

`evaluation/separation_compare/` compares against
[SAME-L](https://github.com/Stability-AI/stable-audio-3),
[DAC](https://github.com/descriptinc/descript-audio-codec) and
[Lin-CAE](https://github.com/lca0/linear_cae); their weights are downloaded
from Hugging Face on first use.

## License

MIT — see `LICENSE`.
