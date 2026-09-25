#!/usr/bin/env python3
"""
Toy Autoencoder Training Script  (GAN + Phase Regularization)
==============================================================

Standalone, lightweight training loop for toy audio autoencoders.
Designed to test the **causal effect of phase regularization** on latent-space
topology and to evaluate linearity properties of the latent manifold.

Supports two model architectures:
  ``--model_type conv``        ToyConvAutoencoder (CNN-based)
  ``--model_type transformer`` ToyTransformerAutoencoder (Transformer, SAME-inspired)

Uses the **SAME loss** (Section 3.1 of the SAME paper) with two configurations:

  ``--use_phase_loss`` **off** (default)
      Loss = Spectral Contrast (Eq. 3) + Adaptive Log-Magnitude (Eq. 4)
      → Phase-blind baseline.

  ``--use_phase_loss`` **on**
      Loss = magnitude terms + IF (Eq. 5) + GD (Eq. 6) + Complex Distance (Eq. 7)
      → Phase-regularised experiment.

Adversarial Training (LSGAN)
-----------------------------
An **STFT-domain PatchGAN discriminator** (``STFTDiscriminator``) is trained
alongside the autoencoder using the Least Squares GAN (LSGAN) objective:

  - **Discriminator**:  ``loss_D = 0.5·MSE(D(real), 1) + 0.5·MSE(D(fake), 0)``
  - **Generator**:      ``loss_G = loss_SAME + λ_adv · MSE(D(fake), 1)``

The adversarial term penalises high-frequency amplitude-modulation artefacts
(whistling) that the STFT-based SAMELoss alone cannot suppress, as described
in the SAME paper.

A **topological evaluation hook** runs every 25 epochs on a fixed validation
subset, measuring encoder/decoder additivity and logging results to WandB.

Data ingestion
--------------
``--data_dir`` accepts one or more dataset roots (e.g. VCTK speech, FSD50K
environmental sounds, FMA-small music).  All ``.wav``/``.flac``/``.mp3`` files
are standardised (mono downmix, digital-silence strip, peak headroom,
resample to ``--sample_rate``) and cached as ``.pt`` tensors in a
per-sample-rate cache directory.

Usage
-----
    # Conv model, phase-blind baseline (no GAN) over one dataset
    python toy_training/train_toy.py \\
        --data_dir /data/wav48_silence_trimmed \\
        --run_name baseline_no_phase \\
        --model_type conv \\
        --sample_rate 16000 \\
        --epochs 100

    # Transformer, phase-regularised + GAN over three aggregated datasets
    python toy_training/train_toy.py \\
        --data_dir /data/wav48_silence_trimmed \\
                   /data/fsd50k/FSD50K.dev_audio \\
                   /data/fma_small \\
        --sample_rate 44100 \\
        --run_name transformer_phase_gan \\
        --model_type transformer \\
        --use_phase_loss \\
        --use_gan \\
        --epochs 100

    # Transformer with explicit final-LayerNorm control: force a final
    # encoder LayerNorm and drop the final decoder LayerNorm.
    python toy_training/train_toy.py \\
        --data_dir /data/wav48_silence_trimmed \\
        --run_name transformer_norm_control \\
        --model_type transformer \\
        --encoder_final_norm \\
        --no_decoder_final_norm \\
        --epochs 100

Dependencies
------------
    torch, torchaudio, wandb, tqdm
    (all available in the project ``.venv``)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

import wandb
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Path wrangling
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_TOY_TRAINING_DIR = _THIS_FILE.parent            # toy_training/
_REPO_ROOT = _TOY_TRAINING_DIR.parent            # stable-audio-3/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_model.autoencoder import ToyConvAutoencoder
from toy_model.discriminator import STFTDiscriminator
from toy_model.transformer_autoencoder import ToyTransformerAutoencoder
from toy_training.losses import (
    SAMELoss,
    cosine_gain_range,
    encoder_cone_loss,
    torres_implicit_linearity_loss,
    universal_mix_loss,
)


# =============================================================================
#  Constants
# =============================================================================

# Audio extensions accepted by the dataset ingestion pipeline.
VALID_AUDIO_EXTS = {".wav", ".flac", ".mp3"}

# Samples whose absolute amplitude is below this threshold are treated as
# digital silence and stripped (leading/trailing).  This replaces the
# speech-specific VAD used by the old VCTK-only pipeline, so that non-speech
# datasets (FSD50K, FMA) are not damaged by a voice-activity detector.
DIGITAL_SILENCE_THRESHOLD = 1e-4

# Peak-normalisation headroom target (95% of full scale).
HEADROOM_PEAK = 0.95

DEFAULT_SAMPLE_RATE = 16000
VAL_FRAC = 0.10          # fraction of tracks reserved for validation
DEFAULT_PREPROCESSED_CACHE_DIR = "/data/preprocessed_cache"


# =============================================================================
#  One-Time Offline Preprocessing  (multi-dataset, sample-rate aware)
# =============================================================================

def _normalise_data_dirs(data_dirs) -> List[str]:
    """Normalise a single path or a list of paths into absolute directories."""
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    dirs: List[str] = []
    for d in data_dirs:
        d = os.path.abspath(os.path.expanduser(str(d)))
        if not os.path.isdir(d):
            raise NotADirectoryError(f"Dataset directory not found: {d}")
        dirs.append(d)
    return dirs


def _collect_audio_files(data_dirs: List[str]) -> List[Tuple[str, str]]:
    """Recursively collect ``(full_path, dataset_root)`` for every audio file.

    Only files whose lower-cased extension is in ``VALID_AUDIO_EXTS`` are kept;
    metadata / non-audio files are skipped.
    """
    files: List[Tuple[str, str]] = []
    for data_dir in data_dirs:
        for dirpath, _dirnames, filenames in os.walk(data_dir):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_AUDIO_EXTS:
                    continue
                files.append((os.path.join(dirpath, fname), data_dir))
    return files


def _strip_digital_silence(
    waveform: torch.Tensor,
    threshold: float = DIGITAL_SILENCE_THRESHOLD,
) -> torch.Tensor:
    """Trim leading/trailing digital silence (samples with |x| < *threshold*).

    Dataset-agnostic: unlike ``torchaudio.functional.vad`` this does not model
    speech, so it is safe for environmental-sound and music datasets.
    """
    if waveform.numel() == 0:
        return waveform
    above = (waveform.abs() > threshold).any(dim=0)
    idx = above.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return waveform[:, :0]  # fully silent
    start, end = int(idx[0, 0]), int(idx[-1, 0])
    return waveform[:, start:end + 1]


def _sanitise_rel_path(rel_path: str) -> str:
    """Turn a relative path into a collision-free cache-filename stem."""
    return rel_path.replace("/", "_").replace("\\", "_")


def preprocess_dataset_cache(
    data_dirs,
    cache_dir: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, object]:
    """Build a standardised, sample-rate-aware cache for one or more datasets.

    Walks every directory in *data_dirs* (a single path or a list), collects
    ``.wav`` / ``.flac`` / ``.mp3`` files, and applies the standardisation
    pipeline to each:

      1. ``torchaudio.load`` (skips corrupt files gracefully),
      2. downmix to mono,
      3. strip absolute digital silence (no speech-specific VAD),
      4. peak-normalise to ``HEADROOM_PEAK`` headroom,
      5. resample to *target_sr*,
      6. save as a float32 ``.pt`` tensor.

    Track IDs are **domain-namespaced** (``<dataset_root_basename>::<track>``)
    so that identically-named directories in different datasets never collide,
    and the manifest records the dataset domain per track.

    If *cache_dir* already contains a ``manifest.json`` with a matching
    ``sample_rate``, preprocessing is skipped and the manifest is reloaded.

    Returns
    -------
    dict  ``{"sample_rate": int, "num_files": int, "num_cached": int,
             "tracks": {track_id: [pt_path, ...]},
             "domains": {track_id: domain}}``
    """
    cache_path = Path(cache_dir)
    manifest_file = cache_path / "manifest.json"

    # ---- Fast path: reuse an existing cache at the same sample rate ----
    if manifest_file.exists():
        with open(manifest_file, "r") as fh:
            manifest = json.load(fh)
        if manifest.get("sample_rate") == target_sr:
            print(f"[Preprocess] Cache found at {cache_dir} "
                  f"(sr={target_sr}) — loading manifest.")
            return manifest
        print(f"[Preprocess] Cache at {cache_dir} has sr="
              f"{manifest.get('sample_rate')} but target is sr={target_sr} "
              f"— rebuilding.")

    cache_path.mkdir(parents=True, exist_ok=True)

    dirs = _normalise_data_dirs(data_dirs)
    all_files = _collect_audio_files(dirs)
    if len(all_files) == 0:
        raise RuntimeError(
            f"No {sorted(VALID_AUDIO_EXTS)} files found under {dirs}"
        )

    print(f"[Preprocess] Found {len(all_files)} audio files across "
          f"{len(dirs)} dataset(s). Building cache (sr={target_sr}) …")

    tracks: Dict[str, List[str]] = defaultdict(list)
    domains: Dict[str, str] = {}

    for src_path, dataset_root in tqdm(all_files, desc="Preprocessing", unit="file"):
        domain = os.path.basename(dataset_root.rstrip(os.sep)) or "dataset"

        # ---- Track / stem derivation (flat or hierarchical layout) ----
        dirpath = os.path.dirname(src_path)
        fname = os.path.basename(src_path)
        stem = os.path.splitext(fname)[0]
        if os.path.normpath(dirpath) == os.path.normpath(dataset_root):
            track_local = stem          # flat layout: one track per file
        else:
            track_local = os.path.basename(dirpath)  # hierarchical: dir = track
        track_id = f"{domain}::{track_local}"

        # ---- Cache filename: domain + sanitised relative path ----
        rel_path = os.path.relpath(src_path, dataset_root)
        rel_stem = os.path.splitext(rel_path)[0]
        cache_name = f"{domain}__{_sanitise_rel_path(rel_stem)}.pt"
        cache_file = cache_path / cache_name

        # ---- 1. Load (skip corrupt files) ----
        try:
            waveform, sr = torchaudio.load(src_path)
        except Exception as exc:
            print(f"  ⚠ Skip {src_path}: load error – {exc}")
            continue

        # ---- 2. Downmix to mono ----
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ---- 3. Strip absolute digital silence ----
        waveform = _strip_digital_silence(waveform)
        if waveform.shape[-1] == 0:
            print(f"  ⚠ Skip {src_path}: silent after trimming")
            continue

        # ---- 4. Peak-normalise with headroom ----
        waveform = waveform / (waveform.abs().max() + 1e-8) * HEADROOM_PEAK

        # ---- 5. Resample to target sample rate ----
        if sr != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=target_sr,
            )

        # ---- 6. Save standardised float32 tensor ----
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(waveform.float(), cache_file)
        tracks[track_id].append(str(cache_file))
        domains[track_id] = domain

    # ---- Write manifest ----
    manifest = {
        "sample_rate": target_sr,
        "num_files": len(all_files),
        "num_cached": sum(len(v) for v in tracks.values()),
        "tracks": dict(tracks),
        "domains": domains,
    }
    with open(manifest_file, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[Preprocess] Done. {manifest['num_cached']} files cached to {cache_dir}")
    return manifest


# =============================================================================
#  Dataset  — preprocessed cache, chunk extraction
# =============================================================================

class AudioPreprocessedDataset(Dataset):
    """PyTorch Dataset over preprocessed ``.pt`` tensor files (any dataset).

    Each ``.pt`` file contains a standardised waveform: mono, digital-silence
    trimmed, peak-normalised, and resampled to the target sample rate.
    ``__getitem__`` only performs rapid disk I/O and chunk extraction —
    **zero DSP computations**, so variable-length clips load without latency.

    Parameters
    ----------
    paths : list[str]
        Full paths to the preprocessed ``.pt`` files in the cache.
    chunk_samples : int
        Number of audio samples per chunk (e.g. ``int(2.0 * 16000)``).
    deterministic : bool
        If True, extract the **centre** chunk (for validation).
        If False, extract a **random** chunk (for training).
    """

    def __init__(
        self,
        paths: List[str],
        chunk_samples: int,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self._paths = list(paths)
        self.chunk_samples = chunk_samples
        self.deterministic = deterministic

        if len(self._paths) == 0:
            raise RuntimeError(
                "No preprocessed .pt files provided to AudioPreprocessedDataset."
            )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load preprocessed tensor from cache, extract chunk.

        Returns
        -------
        torch.Tensor  shape ``(1, chunk_samples)``, dtype float32.
        """
        path = self._paths[idx]

        try:
            waveform = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}")

        # Ensure correct shape: (1, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # ---- Extract chunk (with symmetric zero-padding for short clips) ----
        total_len = waveform.shape[-1]
        if total_len < self.chunk_samples:
            pad_total = self.chunk_samples - total_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
        else:
            if self.deterministic:
                # Centre chunk (stable across epochs)
                start = (total_len - self.chunk_samples) // 2
            else:
                # Random chunk (augmentation)
                start = random.randint(0, total_len - self.chunk_samples)
            waveform = waveform[:, start:start + self.chunk_samples]

        return waveform.float()  # (1, chunk_samples)


# Backwards-compatible alias for the old VCTK-only name.
VCTKPreprocessedDataset = AudioPreprocessedDataset


# =============================================================================
#  Train / Validation split  (track-level, via preprocessed cache)
# =============================================================================

def _build_datasets(
    data_dirs,
    chunk_samples: int,
    cache_dir: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    val_frac: float = VAL_FRAC,
    split_seed: int = 42,
) -> Tuple[AudioPreprocessedDataset, AudioPreprocessedDataset]:
    """Ensure cache exists, group cached clips by (domain-namespaced) track,
    split 90/10 at the track level, and build datasets.

    Track IDs are domain-namespaced by ``preprocess_dataset_cache``, so the
    deterministic split reserves whole tracks across every ingested dataset
    domain (VCTK, FSD50K, FMA, …).

    Returns
    -------
    train_ds : AudioPreprocessedDataset  (random chunks)
    val_ds   : AudioPreprocessedDataset  (centre chunks, deterministic)
    """
    manifest = preprocess_dataset_cache(data_dirs, cache_dir, target_sr=target_sr)
    stems_by_track = manifest.get("tracks", {})
    tracks = sorted(stems_by_track.keys())
    n_tracks = len(tracks)

    if n_tracks < 2:
        raise RuntimeError(
            f"Need at least 2 tracks for a train/val split; got {n_tracks}."
        )

    # Shuffle tracks deterministically
    rng = random.Random(split_seed)
    rng.shuffle(tracks)

    n_val_tracks = max(1, int(n_tracks * val_frac))
    val_tracks = set(tracks[:n_val_tracks])
    train_tracks = set(tracks[n_val_tracks:])

    train_paths: List[str] = []
    val_paths: List[str] = []

    for track, paths in stems_by_track.items():
        if track in val_tracks:
            val_paths.extend(paths)
        else:
            train_paths.extend(paths)

    total_stems = sum(len(v) for v in stems_by_track.values())
    print(
        f"[Data] {n_tracks} tracks, {total_stems} stems  ->  "
        f"train: {len(train_paths)} stems ({len(train_tracks)} tracks),  "
        f"val: {len(val_paths)} stems ({len(val_tracks)} tracks)"
    )

    train_ds = AudioPreprocessedDataset(
        train_paths, chunk_samples, deterministic=False,
    )
    val_ds = AudioPreprocessedDataset(
        val_paths, chunk_samples, deterministic=True,
    )
    return train_ds, val_ds


# =============================================================================
#  Validation loop
# =============================================================================

@torch.no_grad()
# =============================================================================
#  Validation
# =============================================================================

def _encoder_cone_weight_at(
    weight: float,
    progress: float,
    anneal_fraction: float,
) -> float:
    """Effective encoder-cone weight λ at a given training progress.

    With ``anneal_fraction > 0``, λ stays at its full value during the first
    ``1 - anneal_fraction`` of the run and is cosine-annealed to 0 over the
    final ``anneal_fraction`` (coarse-to-fine: learn encoder homomorphism
    first, then let the soft-norm KL / reconstruction re-spread the codes so
    the latent stays diffusion-friendly).  ``progress`` is the fraction of the
    training horizon (e.g. ``(epoch - 1) / epochs``).
    """
    if anneal_fraction <= 0.0 or anneal_fraction >= 1.0:
        return weight                      # anneal disabled
    if progress >= 1.0:
        return 0.0                         # end of the run: λ → 0
    p = (progress - (1.0 - anneal_fraction)) / anneal_fraction
    p = max(0.0, min(1.0, p))
    return weight * 0.5 * (1.0 + math.cos(math.pi * p))


def run_validation(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: SAMELoss,
    use_phase: bool,
    device: torch.device,
    use_latent_mixup: bool = False,
    mixup_weight: float = 1.0,
    use_implicit_linearity: bool = False,
    implicit_linearity_weight: float = 1.0,
    implicit_gain_min: float = 0.0,
    implicit_gain_max: float = 3.0,
    implicit_gain_prob: float = 0.8,
    implicit_mix_fraction: float = 0.5,
    no_implicit_gain_anneal: bool = False,
    implicit_progress: float = 1.0,
    use_universal_mix: bool = False,
    universal_mix_weight: float = 1.0,
    universal_gain_min: float = 0.0,
    universal_gain_max: float = 3.0,
    universal_gain_prob: float = 0.8,
    universal_mix_fraction: float = 0.5,
    no_universal_gain_anneal: bool = False,
    universal_progress: float = 1.0,
    universal_sampler: str = "polar",
    use_encoder_cone: bool = False,
    encoder_cone_weight: float = 1.0,
    encoder_cone_mode: str = "auto",
    encoder_cone_anneal_fraction: float = 0.0,
    encoder_cone_progress: float = 1.0,
) -> Dict[str, float]:
    """Evaluate model on the validation set.

    When ``use_latent_mixup`` is True, the forward pass is split into
    encode/decode and the trigonometric in-batch-rolling mixup loss is
    computed and added to the total, mirroring the training ``total_G_loss``
    exactly so that train/val curves remain comparable.

    When ``use_implicit_linearity`` is True, the Torres-style implicit
    homogeneity/additivity regularizer is evaluated the same way.

    When ``use_universal_mix`` is True, the universal positive-weight
    interpolation (constant-power cone) loss is evaluated the same way.

    When ``use_encoder_cone`` is True, the encoder-side cone-consistency
    (homomorphism) loss is evaluated the same way.  ``encoder_cone_mode`` is
    "auto" (universal cone if ``use_universal_mix`` else constant-power trig
    arc), "trig" or "universal".

    Returns
    -------
    dict with keys ``val_total``, ``val_mag``, ``val_phase``, ``val_mixup``,
    ``val_implicit``, ``val_univ``, ``val_enc_cone``.
    """
    model.eval()

    total_loss = 0.0
    mag_loss = 0.0
    phase_loss = 0.0
    mixup_loss_total = 0.0
    implicit_loss_total = 0.0
    univ_loss_total = 0.0
    enc_cone_loss_total = 0.0
    n_batches = 0

    for batch in dataloader:
        audio = batch.to(device)                     # (B, 1, chunk_samples)

        # ---- Forward pass: split encode/decode when an auxiliary latent loss is active ----
        if (use_latent_mixup or use_implicit_linearity or use_universal_mix
                or use_encoder_cone):
            z = model.encode(audio)                                    # (B, L, latent_dim)
            reconstructed = model.decode(z, original_length=audio.shape[-1])
        else:
            reconstructed = model(audio)                               # (B, 1, chunk_samples)

        t_loss, m_loss = loss_fn(reconstructed, audio, use_phase=use_phase, return_components=True)
        p_loss = t_loss - m_loss

        # ---- Latent Mixup (Trigonometric, in-batch rolling) ----
        if use_latent_mixup:
            alpha = torch.rand(audio.shape[0], 1, 1, device=device)    # (B, 1, 1)
            audio_rolled = torch.roll(audio, shifts=1, dims=0)         # (B, 1, T)
            z_rolled = torch.roll(z, shifts=1, dims=0)                 # (B, L, D)

            theta = alpha * (math.pi / 2.0)
            z_mix = torch.cos(theta) * z + torch.sin(theta) * z_rolled
            audio_mix_target = torch.cos(theta) * audio + torch.sin(theta) * audio_rolled

            recon_mix = model.decode(z_mix, original_length=audio.shape[-1])
            loss_mixup = loss_fn(
                recon_mix, audio_mix_target,
                use_phase=use_phase,
            )
            t_loss = t_loss + mixup_weight * loss_mixup
            mixup_loss_total += loss_mixup.item()

        # ---- Torres-style implicit linearity (homogeneity + additivity) ----
        if use_implicit_linearity:
            loss_implicit = torres_implicit_linearity_loss(
                model, loss_fn, audio, z, device,
                gain_min=implicit_gain_min,
                gain_max=implicit_gain_max,
                gain_prob=implicit_gain_prob,
                mix_fraction=implicit_mix_fraction,
                use_phase=use_phase,
                progress=implicit_progress,
                anneal_gain=not no_implicit_gain_anneal,
            )
            t_loss = t_loss + implicit_linearity_weight * loss_implicit
            implicit_loss_total += loss_implicit.item()

        # ---- Universal positive-weight interpolation (constant-power cone) ----
        if use_universal_mix:
            loss_univ = universal_mix_loss(
                model, loss_fn, audio, z, device,
                gain_min=universal_gain_min,
                gain_max=universal_gain_max,
                gain_prob=universal_gain_prob,
                mix_fraction=universal_mix_fraction,
                use_phase=use_phase,
                progress=universal_progress,
                anneal_gain=not no_universal_gain_anneal,
                sampler=universal_sampler,
            )
            t_loss = t_loss + universal_mix_weight * loss_univ
            univ_loss_total += loss_univ.item()

        # ---- Encoder-side cone consistency (homomorphism) ----
        if use_encoder_cone:
            mode = encoder_cone_mode
            if mode == "auto":
                mode = "universal" if use_universal_mix else "trig"
            if mode == "universal":
                loss_enc_cone = encoder_cone_loss(
                    model, audio, device,
                    gain_min=universal_gain_min,
                    gain_max=universal_gain_max,
                    gain_prob=universal_gain_prob,
                    mix_fraction=universal_mix_fraction,
                    progress=universal_progress,
                    anneal_gain=not no_universal_gain_anneal,
                )
            else:  # constant-power trig arc, whole batch
                loss_enc_cone = encoder_cone_loss(model, audio, device, trig_only=True)
            enc_w = _encoder_cone_weight_at(
                encoder_cone_weight, encoder_cone_progress, encoder_cone_anneal_fraction,
            )
            t_loss = t_loss + enc_w * loss_enc_cone
            enc_cone_loss_total += loss_enc_cone.item()

        total_loss += t_loss.item()
        mag_loss   += m_loss.item()
        phase_loss += p_loss.item() if use_phase else 0.0
        n_batches  += 1

    model.train()
    return {
        "val_total": total_loss / n_batches,
        "val_mag":   mag_loss   / n_batches,
        "val_phase": phase_loss / n_batches,
        "val_mixup": mixup_loss_total / n_batches if use_latent_mixup else 0.0,
        "val_implicit": implicit_loss_total / n_batches if use_implicit_linearity else 0.0,
        "val_univ": univ_loss_total / n_batches if use_universal_mix else 0.0,
        "val_enc_cone": enc_cone_loss_total / n_batches if use_encoder_cone else 0.0,
    }


# =============================================================================
#  Early Stopping
# =============================================================================

class EarlyStopping:
    """Track the best validation metric and stop when no improvement is seen
    for ``patience`` consecutive checks.

    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    min_delta : float
        Minimum absolute improvement to count as "better".
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        """Update state with the current validation loss.

        Returns
        -------
        bool  True if this epoch achieved a new best score.
        """
        if self.best_score is None:
            self.best_score = val_loss
            return True  # first epoch -> save

        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            return True  # improved -> save
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# =============================================================================
#  Checkpoint helpers
# =============================================================================

def _checkpoint_dir() -> Path:
    d = Path(_TOY_TRAINING_DIR) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(
    model: nn.Module,
    run_name: str,
    optimizer_G: torch.optim.Optimizer,
    scheduler_G: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    discriminator: nn.Module | None = None,
    optimizer_D: torch.optim.Optimizer | None = None,
    scheduler_D: torch.optim.lr_scheduler.LRScheduler | None = None,
    model_config: dict | None = None,
) -> str:
    """Save full training state so we can resume after interruption."""
    out_path = _checkpoint_dir() / f"{run_name}.pt"
    ckpt: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "epoch": epoch,
    }
    if model_config is not None:
        ckpt["model_config"] = model_config
    if scheduler_G is not None:
        ckpt["scheduler_G_state_dict"] = scheduler_G.state_dict()
    if discriminator is not None:
        ckpt["discriminator_state_dict"] = discriminator.state_dict()
    if optimizer_D is not None:
        ckpt["optimizer_D_state_dict"] = optimizer_D.state_dict()
    if scheduler_D is not None:
        ckpt["scheduler_D_state_dict"] = scheduler_D.state_dict()
    torch.save(ckpt, out_path)
    print(f"[Checkpoint] Saved to {out_path}")
    return str(out_path)


def save_best_checkpoint(
    model: nn.Module,
    run_name: str,
    optimizer_G: torch.optim.Optimizer,
    scheduler_G: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    discriminator: nn.Module | None = None,
    optimizer_D: torch.optim.Optimizer | None = None,
    scheduler_D: torch.optim.lr_scheduler.LRScheduler | None = None,
    model_config: dict | None = None,
) -> str:
    """Save best-model training state so we can resume after interruption."""
    out_path = _checkpoint_dir() / f"{run_name}_best.pt"
    ckpt: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "epoch": epoch,
    }
    if model_config is not None:
        ckpt["model_config"] = model_config
    if scheduler_G is not None:
        ckpt["scheduler_G_state_dict"] = scheduler_G.state_dict()
    if discriminator is not None:
        ckpt["discriminator_state_dict"] = discriminator.state_dict()
    if optimizer_D is not None:
        ckpt["optimizer_D_state_dict"] = optimizer_D.state_dict()
    if scheduler_D is not None:
        ckpt["scheduler_D_state_dict"] = scheduler_D.state_dict()
    torch.save(ckpt, out_path)
    print(f"[Checkpoint] Best model saved to {out_path}")
    return str(out_path)


# =============================================================================
#  Soft-Norm KL Regularisation Helper
# =============================================================================

def _compute_soft_norm_kl(z_norm: torch.Tensor) -> torch.Tensor:
    """Dual-axis KL-like regularisation loss (SAME Eq. 1).

    Computed on the clean normalised latent ``z_norm`` of shape
    ``(B, L, D)``.  Penalises deviation from zero-mean, unit-variance
    statistics along the time (sequence) and channel axes independently.

    Parameters
    ----------
    z_norm : torch.Tensor  shape ``(B, L, D)``

    Returns
    -------
    torch.Tensor  scalar KL loss.
    """
    B, L, D = z_norm.shape
    eps = 1e-8

    # ---- Time-axis statistics (per channel, over sequence length) ----
    mu_t = z_norm.mean(dim=1)               # (B, D)
    sigma_t_sq = z_norm.var(dim=1, unbiased=False)  # (B, D)
    loss_kl_t = (mu_t ** 2 + sigma_t_sq - torch.log(sigma_t_sq + eps) - 1.0).mean()

    # ---- Channel-axis statistics (per timestep, over channels) ----
    mu_c = z_norm.mean(dim=2)               # (B, L)
    sigma_c_sq = z_norm.var(dim=2, unbiased=False)  # (B, L)
    loss_kl_c = (mu_c ** 2 + sigma_c_sq - torch.log(sigma_c_sq + eps) - 1.0).mean()

    # ---- Combined (channel term downweighted by 0.4 per SAME Eq. 1) ----
    return loss_kl_t + 0.4 * loss_kl_c


# =============================================================================
#  Main Training Loop
# =============================================================================

def train(args: argparse.Namespace) -> None:
    """Run the full training pipeline with validation and early stopping."""

    # ---- Validate sample rate ----
    if args.sample_rate <= 0:
        raise ValueError(
            f"--sample_rate must be a positive integer, got {args.sample_rate}"
        )

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- Device ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    print(f"[Data] Sample rate: {args.sample_rate} Hz")

    # ---- Datasets & DataLoaders ----
    # Embed the sample rate into the cache directory so that caches built at
    # different rates never collide (e.g. sr16000 vs sr44100).
    cache_dir = os.path.join(args.preprocess_cache_dir, f"sr{args.sample_rate}")
    chunk_samples = int(args.chunk_seconds * args.sample_rate)
    train_ds, val_ds = _build_datasets(
        args.data_dirs, chunk_samples,
        cache_dir=cache_dir,
        target_sr=args.sample_rate,
        val_frac=args.val_frac, split_seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Optional per-epoch batch cap (fast hyperparameter iteration) ----
    train_steps_per_epoch = len(train_loader)
    if args.limit_train_batches > 0:
        train_steps_per_epoch = min(len(train_loader), args.limit_train_batches)
        print(f"[Data] Capping training to {train_steps_per_epoch} batches/epoch "
              f"(full pass = {len(train_loader)} batches).")

    # ---- Model (with dropout regularisation) ----
    # The final encoder/decoder LayerNorm can be toggled via the
    # --encoder_final_norm / --decoder_final_norm CLI flags.  When a flag is
    # left unspecified (None), the auto behaviour applies: latent mixup and the
    # Torres-style implicit linearity loss both require a scale-sensitive
    # bottleneck, so the corresponding final norm is disabled then (preserving
    # energy differences between trig-interpolated latent paths and between
    # gain-scaled / summed latents).
    use_scale_sensitive = (
        (args.use_latent_mixup or args.use_implicit_linearity or args.use_universal_mix)
        and args.model_type == "transformer"
    )
    encoder_final_norm = (
        not use_scale_sensitive
        if args.encoder_final_norm is None
        else args.encoder_final_norm
    )
    decoder_final_norm = (
        not use_scale_sensitive
        if args.decoder_final_norm is None
        else args.decoder_final_norm
    )
    model_config: dict = {}
    if args.model_type == "transformer":
        # Hybrid activation (SAME §2.2): base activation everywhere except the
        # final `decoder_num_snake_layers` decoder layers, which use Snake.
        # Legacy `--activation snake` (global periodic) is mapped to the hybrid
        # equivalent: base leaky_relu + Snake on all decoder layers.
        if args.activation == "snake":
            print("[Model] ⚠ Global 'snake' activation is deprecated per SAME "
                  "§2.2 — using base 'leaky_relu' with Snake on ALL decoder "
                  "layers (decoder_num_snake_layers=num_layers).")
            act, snake_layers = "leaky_relu", 6
        else:
            act, snake_layers = args.activation, args.decoder_num_snake_layers
        model_config = {
            "model_type": "transformer",
            "activation": act,
            "decoder_num_snake_layers": snake_layers,
            "encoder_final_norm": encoder_final_norm,
            "decoder_final_norm": decoder_final_norm,
            "use_soft_norm": args.use_soft_norm,
            "use_universal_mix": args.use_universal_mix,
            "latent_dim": args.latent_dim,
            "patch_size": args.patch_size,
            "d_model": args.d_model,
        }
        model = ToyTransformerAutoencoder(
            latent_dim=args.latent_dim,
            d_model=args.d_model,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=args.dropout,
            patch_size=args.patch_size,
            activation=act,
            decoder_num_snake_layers=snake_layers,
            encoder_final_norm=encoder_final_norm,
            decoder_final_norm=decoder_final_norm,
            use_soft_norm=args.use_soft_norm,
        ).to(device)
    else:
        model = ToyConvAutoencoder(
            latent_dim=64,
            dropout=args.dropout,
            activation=args.activation,
        ).to(device)
    print(f"[Model] Type: {args.model_type}")
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Activation: {args.activation}")
    if args.model_type == "transformer":
        print(f"[Model] Decoder Snake layers: {model_config['decoder_num_snake_layers']}"
              f" (final {model_config['decoder_num_snake_layers']} decoder layers)")
        print(f"[Model] Encoder final LayerNorm: "
              f"{'ON' if encoder_final_norm else 'OFF'}")
        print(f"[Model] Decoder final LayerNorm: "
              f"{'ON' if decoder_final_norm else 'OFF'}")
    print(f"[Model] Dropout: p={args.dropout}")
    if use_scale_sensitive:
        print(f"[Model] 🔬 Scale-sensitive bottleneck active (latent mixup): "
              f"final norms default to OFF unless overridden via CLI.")
    if args.use_soft_norm:
        print(f"[Model] 🧊 SAME Soft-Norm Bottleneck enabled "
              f"(EMA std tracking, noise injection, KL weight="
              f"{args.soft_norm_kl_weight}).")

    # ---- Gain-anneal sanity note for the implicit / universal losses ----
    if (args.use_implicit_linearity or args.use_universal_mix) and args.epochs > 2000:
        print("[Model] ⚠ Gain-anneal horizon: the implicit/universal gain "
              "range is cosine-annealed over --epochs via progress="
              "(epoch-1)/epochs. With a large safety-ceiling --epochs"
              f"={args.epochs}, runs that early-stop near epoch ~{args.patience * 12} "
              "never leave the progress≈0 endpoint of the ramp: with the "
              "corrected direction that is the FULL (gain_min, gain_max) cone "
              "for the whole run, with NO final convergence to the "
              "constant-power (1,1) regime. If you want the cone→constant-power "
              "curriculum, set --epochs close to the planned run length (e.g. "
              "250–500); otherwise pass --no_universal_gain_anneal / "
              "--no_implicit_gain_anneal for an explicit fixed range. The "
              "per-epoch effective range is logged below.")

    # ---- Discriminator (GAN / Morph GAN) ----
    discriminator = None
    if args.use_gan or args.use_morph_gan:
        discriminator = STFTDiscriminator(
            n_fft=1024,
            hop_length=256,
            hidden_channels=32,
        ).to(device)
        print(f"[Discriminator] Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ---- Loss ----
    loss_fn = SAMELoss(eps=1e-8).to(device)
    mse_loss = nn.MSELoss().to(device)

    # ---- Optimisers (AdamW with weight decay) ----
    optimizer_G = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    optimizer_D: Optional[torch.optim.AdamW] = None
    scheduler_G: Optional[torch.optim.lr_scheduler.OneCycleLR] = None
    scheduler_D: Optional[torch.optim.lr_scheduler.OneCycleLR] = None

    if args.use_gan or args.use_morph_gan:
        optimizer_D = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # ---- Schedulers ----
    # 'onecycle': OneCycleLR with warmup — anneals LR to ~0 over exactly
    #   `epochs` steps (fixed-horizon schedule).
    # 'plateau': ReduceLROnPlateau — decays LR when val stops improving, so the
    #   LR is low/converged by the time early stopping fires. Recommended for
    #   "run until early stopping" full training.
    if args.scheduler == "onecycle":
        scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_G,
            max_lr=args.lr,
            steps_per_epoch=train_steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.05,
        )
        if args.use_gan or args.use_morph_gan:
            scheduler_D = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_D,
                max_lr=args.lr,
                steps_per_epoch=train_steps_per_epoch,
                epochs=args.epochs,
                pct_start=0.05,
            )
    else:  # "plateau"
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_G, mode="min", factor=0.5, patience=5,
            threshold=1e-4, cooldown=0, min_lr=1e-6,
        )
        if args.use_gan or args.use_morph_gan:
            scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_D, mode="min", factor=0.5, patience=5,
                threshold=1e-4, cooldown=0, min_lr=1e-6,
            )

    # ---- Early stopping ----
    early_stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
    )

    # ---- Resume from checkpoint ----
    start_epoch = 1
    if args.resume_checkpoint is not None:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {ckpt_path}"
            )
        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Backward-compatible key detection (old checkpoints used "model", new ones use "model_state_dict")
        model_key = "model_state_dict" if "model_state_dict" in checkpoint else "model"
        opt_G_key = "optimizer_G_state_dict" if "optimizer_G_state_dict" in checkpoint else None
        opt_D_key = "optimizer_D_state_dict" if "optimizer_D_state_dict" in checkpoint else None
        sched_G_key = "scheduler_G_state_dict" if "scheduler_G_state_dict" in checkpoint else None
        sched_D_key = "scheduler_D_state_dict" if "scheduler_D_state_dict" in checkpoint else None
        disc_key = "discriminator_state_dict" if "discriminator_state_dict" in checkpoint else ("discriminator" if "discriminator" in checkpoint else None)

        # Restore model weights
        model.load_state_dict(checkpoint[model_key])

        # Restore optimizer states (only if available — old checkpoints lack them)
        if opt_G_key is not None:
            optimizer_G.load_state_dict(checkpoint[opt_G_key])
        if args.use_gan and opt_D_key is not None:
            optimizer_D.load_state_dict(checkpoint[opt_D_key])

        # Restore scheduler states
        if sched_G_key is not None:
            scheduler_G.load_state_dict(checkpoint[sched_G_key])
        if args.use_gan and scheduler_D is not None and sched_D_key is not None:
            scheduler_D.load_state_dict(checkpoint[sched_D_key])

        # Restore discriminator if present
        if args.use_gan and disc_key is not None:
            discriminator.load_state_dict(checkpoint[disc_key])

        # Resume epoch counter (old checkpoints may lack the "epoch" key)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        else:
            # Old-format checkpoint: we don't know the epoch — start from 1
            start_epoch = 1
            print("[Resume] Old-format checkpoint without epoch info; starting from epoch 1.")
        print(f"[Resume] Continuing from epoch {start_epoch} / {args.epochs}")

        # Reset early-stopping counter to avoid immediate trigger on restart
        early_stopper.counter = 0
        early_stopper.early_stop = False

    # ---- WandB ----
    wandb_kwargs: dict = {
        "project": "toy_ae_phase",
        "name": args.run_name,
        "config": vars(args),
        "reinit": True,
    }
    if args.wandb_run_id is not None:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)
    wandb.watch(model, log="gradients", log_freq=100)

    # ---- Topological Evaluation: fixed validation pairs ----
    # Isolate 16 *pairs* (32 clips) from the validation set deterministically
    # for live additivity checks every 25 epochs.
    TOPO_PAIRS = 16
    TOPO_INTERVAL = 25
    if not args.fast_eval:
        topo_rng = torch.Generator().manual_seed(args.seed + 999)
        val_indices = torch.randperm(len(val_ds), generator=topo_rng)[:TOPO_PAIRS * 2]
        topo_audio_a_batch = torch.stack(
            [val_ds[i.item()] for i in val_indices[0::2]], dim=0,
        ).to(device)  # (TOPO_PAIRS, 1, chunk_samples)
        topo_audio_b_batch = torch.stack(
            [val_ds[i.item()] for i in val_indices[1::2]], dim=0,
        ).to(device)

        # ---- Audio samples directory for inspection ----
        topo_samples_dir = _checkpoint_dir() / "topo_samples" / args.run_name
        topo_samples_dir.mkdir(parents=True, exist_ok=True)


    def _run_topological_eval(
        mdl: nn.Module,
        audio_a: torch.Tensor,
        audio_b: torch.Tensor,
    ) -> dict[str, float]:
        """Run additivity and latent-geometry tests; return scalar metrics.

        Returns
        -------
        dict with keys:
            ``decoder_additivity_cosine_sim``
            ``encoder_additivity_mse``
            ``latent_density``
            ``latent_convexity``
            ``Decoded_A`` (tensor, first pair)
            ``Decoded_B`` (tensor, first pair)
            ``Decoded_Mix`` (tensor, first pair)
        """
        # --- Encode individuals ---
        Z_A = mdl.encode(audio_a)                     # (N, L, latent_dim)
        Z_B = mdl.encode(audio_b)

        # --- Metric A: Decoder Additivity ---
        Z_mid = 0.5 * Z_A + 0.5 * Z_B
        Decoded_Mix = mdl.decode(Z_mid)               # (N, 1, T)
        Decoded_A = mdl.decode(Z_A)
        Decoded_B = mdl.decode(Z_B)
        Ideal_Mix = 0.5 * Decoded_A + 0.5 * Decoded_B

        # Flatten over batch & time for cosine similarity
        dm_flat = Decoded_Mix.reshape(-1)
        im_flat = Ideal_Mix.reshape(-1)
        cosine_sim = F.cosine_similarity(
            dm_flat.unsqueeze(0), im_flat.unsqueeze(0),
        ).item()

        # --- Metric B: Encoder Additivity ---
        Audio_Mix = 0.5 * audio_a + 0.5 * audio_b      # (N, 1, T)
        Z_mix_encoded = mdl.encode(Audio_Mix)           # (N, L, latent_dim)
        mse_enc = F.mse_loss(Z_mix_encoded, Z_mid).item()

        # --- Metric C: Latent Density ---
        # Average pairwise L2 distance among all real latent vectors
        # (pooled from Z_A and Z_B).
        Z_real_pool = torch.cat([Z_A, Z_B], dim=0)        # (2N, L, latent_dim)
        # Flatten to (2N, L·latent_dim) for pairwise distances
        Z_flat = Z_real_pool.reshape(Z_real_pool.shape[0], -1)  # (2N, D)
        # Pairwise distances via pdist (more efficient than manual loops)
        pairwise_dists = torch.cdist(Z_flat, Z_flat, p=2)  # (2N, 2N)
        # Exclude self-distances: mask the diagonal
        mask = ~torch.eye(pairwise_dists.shape[0], dtype=torch.bool, device=pairwise_dists.device)
        latent_density = pairwise_dists[mask].mean().item()

        # --- Metric D: Latent Convexity ---
        # For each interpolated point in Z_mid, find distance to nearest
        # real latent (excluding its own endpoints Z_A[i] and Z_B[i]).
        Z_mid_flat = Z_mid.reshape(Z_mid.shape[0], -1)     # (N, D)
        N = Z_mid.shape[0]
        convexity_dists = []
        for i in range(N):
            # Exclude Z_A[i] (index i) and Z_B[i] (index i+N) from the pool
            exclude_mask = torch.ones(Z_flat.shape[0], dtype=torch.bool, device=Z_flat.device)
            exclude_mask[i] = False       # Z_A[i]
            exclude_mask[i + N] = False   # Z_B[i]
            # Distances from this midpoint to all non-excluded real latents
            dists_i = torch.cdist(
                Z_mid_flat[i:i+1], Z_flat[exclude_mask], p=2,
            )
            convexity_dists.append(dists_i.min().item())
        latent_convexity = float(
            sum(convexity_dists) / len(convexity_dists)
        )

        return {
            "decoder_additivity_cosine_sim": float(cosine_sim),
            "encoder_additivity_mse": float(mse_enc),
            "latent_density": float(latent_density),
            "latent_convexity": latent_convexity,
            "Decoded_A": Decoded_A,
            "Decoded_B": Decoded_B,
            "Decoded_Mix": Decoded_Mix,
        }

    # ---- Training Loop ----
    model.train()
    if discriminator is not None:
        discriminator.train()

    for epoch in range(start_epoch, args.epochs + 1):
        # ================================================================
        #  TRAIN
        # ================================================================
        epoch_train_loss     = 0.0
        epoch_train_mag      = 0.0
        epoch_train_phase    = 0.0
        epoch_train_loss_D   = 0.0
        epoch_train_loss_G_adv = 0.0
        epoch_train_mixup    = 0.0
        epoch_train_implicit = 0.0
        epoch_train_univ     = 0.0
        epoch_train_morph_adv = 0.0
        epoch_train_cycle    = 0.0
        epoch_train_soft_norm_kl = 0.0
        epoch_train_enc_cone  = 0.0
        n_train_batches      = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:4d}/{args.epochs} [train]",
            leave=False,
        )
        for batch_idx, batch in enumerate(pbar):
            if args.limit_train_batches > 0 and batch_idx >= args.limit_train_batches:
                break
            audio = batch.to(device)                     # (B, 1, chunk_samples)

            # ---- Forward pass: split encode/decode when mixup or morph is active ----
            if (args.use_latent_mixup or args.use_implicit_linearity
                    or args.use_universal_mix or args.use_morph_gan
                    or args.use_encoder_cone or args.use_cycle):
                z = model.encode(audio)                                    # (B, L, latent_dim)
                reconstructed = model.decode(z, original_length=audio.shape[-1])
            else:
                reconstructed = model(audio)                               # (B, 1, chunk_samples)

            # ============================================================
            #  DISCRIMINATOR STEP  (LSGAN + Morph negatives)
            # ============================================================
            if args.use_gan or args.use_morph_gan:
                # Real audio → D should output 1
                _feats_real, D_real = discriminator(audio)
                # Fake (detached recon) → D should output 0
                _feats_fake, D_fake_recon = discriminator(reconstructed.detach())

                loss_D_real = mse_loss(D_real, torch.ones_like(D_real))
                loss_D_fake_recon = mse_loss(D_fake_recon, torch.zeros_like(D_fake_recon))
                loss_D = 0.5 * loss_D_real + 0.5 * loss_D_fake_recon
                n_D_terms = 2  # denominator for averaging

                # ---- Morph GAN negatives: teach D that mixes & morphs are fake ----
                if args.use_morph_gan:
                    # Real audio mix (additive) → D should output 0
                    audio_rolled = torch.roll(audio, shifts=1, dims=0)
                    audio_mix_real = 0.5 * audio + 0.5 * audio_rolled
                    _feats_mix, D_fake_mix = discriminator(audio_mix_real)
                    loss_D_fake_mix = mse_loss(D_fake_mix, torch.zeros_like(D_fake_mix))

                    # LERP morph audio (detached) → D should output 0
                    z_rolled = torch.roll(z, shifts=1, dims=0)
                    z_lerp = 0.5 * z + 0.5 * z_rolled
                    audio_morph_detached = model.decode(
                        z_lerp, original_length=audio.shape[-1],
                    ).detach()
                    _feats_morph, D_fake_morph = discriminator(audio_morph_detached)
                    loss_D_fake_morph = mse_loss(D_fake_morph, torch.zeros_like(D_fake_morph))

                    loss_D = loss_D + loss_D_fake_mix + loss_D_fake_morph
                    n_D_terms += 2

                # Average over all terms
                loss_D = loss_D / n_D_terms

                optimizer_D.zero_grad()
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
                if args.scheduler == "onecycle":
                    scheduler_D.step()

                epoch_train_loss_D += loss_D.item()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # ============================================================
            #  GENERATOR STEP  (SAME reconstruction + adversarial)
            # ============================================================
            # ---- SAME Loss (magnitude ± phase) ----
            loss_SAME, mag_raw = loss_fn(
                reconstructed, audio,
                use_phase=args.use_phase_loss,
                return_components=True,
            )
            mag_loss = mag_raw.detach()
            phase_loss_val = loss_SAME - mag_loss

            # ---- Adversarial Loss ----
            if args.use_gan:
                _feats_G, D_fake_for_G = discriminator(reconstructed)
                loss_adv = mse_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))
                total_G_loss = loss_SAME + args.adv_loss_weight * loss_adv
                epoch_train_loss_G_adv += loss_adv.item()
            else:
                total_G_loss = loss_SAME
                loss_adv = torch.tensor(0.0, device=device)

            # ---- SAME Soft-Norm KL Regularisation ----
            loss_soft_norm_kl = torch.tensor(0.0, device=device)
            if args.use_soft_norm and model.soft_norm is not None:
                z_clean = model.soft_norm.z_norm_clean
                if z_clean is not None:
                    loss_soft_norm_kl = _compute_soft_norm_kl(z_clean)
                    total_G_loss = total_G_loss + args.soft_norm_kl_weight * loss_soft_norm_kl
                    epoch_train_soft_norm_kl += loss_soft_norm_kl.item()

            # ---- Latent Mixup (Trigonometric, in-batch rolling) ----
            loss_mixup = torch.tensor(0.0, device=device)
            if args.use_latent_mixup:
                # In-batch rolling: pair each sample with its neighbour
                # to avoid a 2× compute cost while still providing diverse
                # mixup pairs every batch.
                alpha = torch.rand(audio.shape[0], 1, 1, device=device)    # (B, 1, 1)
                audio_rolled = torch.roll(audio, shifts=1, dims=0)         # (B, 1, T)
                z_rolled = torch.roll(z, shifts=1, dims=0)                 # (B, L, D)

                # Constant-power trigonometric mixing law
                theta = alpha * (math.pi / 2.0)
                z_mix = torch.cos(theta) * z + torch.sin(theta) * z_rolled
                audio_mix_target = torch.cos(theta) * audio + torch.sin(theta) * audio_rolled

                # One extra decoder pass for the mixup
                recon_mix = model.decode(z_mix, original_length=audio.shape[-1])
                loss_mixup = loss_fn(
                    recon_mix, audio_mix_target,
                    use_phase=args.use_phase_loss,
                )
                total_G_loss = total_G_loss + args.mixup_weight * loss_mixup
                epoch_train_mixup += loss_mixup.item()

            # ---- Torres-style implicit linearity (homogeneity + additivity) ----
            loss_implicit = torch.tensor(0.0, device=device)
            if args.use_implicit_linearity:
                # Cosine gain-range anneal over the configured training horizon:
                # (gain_min, gain_max) -> (1, 1) as epoch -> epochs (paper: 0-3 -> 1).
                progress = (epoch - 1) / args.epochs if args.epochs > 0 else 1.0
                loss_implicit = torres_implicit_linearity_loss(
                    model, loss_fn, audio, z, device,
                    gain_min=args.implicit_gain_min,
                    gain_max=args.implicit_gain_max,
                    gain_prob=args.implicit_gain_prob,
                    mix_fraction=args.implicit_mix_fraction,
                    use_phase=args.use_phase_loss,
                    progress=progress,
                    anneal_gain=not args.no_implicit_gain_anneal,
                )
                total_G_loss = total_G_loss + args.implicit_linearity_weight * loss_implicit
                epoch_train_implicit += loss_implicit.item()

            # ---- Universal positive-weight interpolation (constant-power cone) ----
            loss_univ = torch.tensor(0.0, device=device)
            if args.use_universal_mix:
                # Cosine gain-range anneal over the configured training horizon:
                # (gain_min, gain_max) -> (1, 1) as epoch -> epochs.
                progress = (epoch - 1) / args.epochs if args.epochs > 0 else 1.0
                loss_univ = universal_mix_loss(
                    model, loss_fn, audio, z, device,
                    gain_min=args.universal_gain_min,
                    gain_max=args.universal_gain_max,
                    gain_prob=args.universal_gain_prob,
                    mix_fraction=args.universal_mix_fraction,
                    use_phase=args.use_phase_loss,
                    progress=progress,
                    anneal_gain=not args.no_universal_gain_anneal,
                    sampler=args.universal_sampler,
                )
                total_G_loss = total_G_loss + args.universal_mix_weight * loss_univ
                epoch_train_univ += loss_univ.item()

            # ---- Encoder-side cone consistency (homomorphism) ----
            # MSE(E(w1·A + w2·B), w1·E(A) + w2·E(B)) on clean soft-norm codes —
            # the encoder mirror of the decoder cone losses.  Sampler follows
            # the active decoder loss: the full universal cone when
            # --use_universal_mix, else the constant-power trig arc.
            loss_enc_cone = torch.tensor(0.0, device=device)
            if args.use_encoder_cone:
                enc_mode = args.encoder_cone_mode
                if enc_mode == "auto":
                    enc_mode = "universal" if args.use_universal_mix else "trig"
                if enc_mode == "universal":
                    progress = (epoch - 1) / args.epochs if args.epochs > 0 else 1.0
                    loss_enc_cone = encoder_cone_loss(
                        model, audio, device,
                        gain_min=args.universal_gain_min,
                        gain_max=args.universal_gain_max,
                        gain_prob=args.universal_gain_prob,
                        mix_fraction=args.universal_mix_fraction,
                        progress=progress,
                        anneal_gain=not args.no_universal_gain_anneal,
                    )
                else:  # constant-power trig arc, whole batch
                    loss_enc_cone = encoder_cone_loss(
                        model, audio, device, trig_only=True,
                    )
                enc_w = _encoder_cone_weight_at(
                    args.encoder_cone_weight,
                    (epoch - 1) / args.epochs if args.epochs > 0 else 1.0,
                    args.encoder_cone_anneal_fraction,
                )
                total_G_loss = total_G_loss + enc_w * loss_enc_cone
                epoch_train_enc_cone += loss_enc_cone.item()

            # ---- Morph GAN: LERP feature morphing + Cycle Consistency ----
            loss_morph_adv = torch.tensor(0.0, device=device)
            loss_cycle = torch.tensor(0.0, device=device)
            if args.use_morph_gan:
                # In-batch rolling for LERP pairs
                z_rolled_morph = torch.roll(z, shifts=1, dims=0)           # (B, L, D)
                z_lerp = 0.5 * z + 0.5 * z_rolled_morph                    # (B, L, D)

                # Decode the LERP latent → morph audio
                audio_morph = model.decode(
                    z_lerp, original_length=audio.shape[-1],
                )                                                           # (B, 1, T)

                # ---- Morph Adversarial Loss (Generator) ----
                _feats_morph_G, D_morph_G = discriminator(audio_morph)
                loss_morph_adv = mse_loss(
                    D_morph_G, torch.ones_like(D_morph_G),
                )

                # ---- Cycle Consistency Loss ----
                # Freeze encoder: gradients flow through decoder only.
                # Transformer: freeze encoder_transformer.
                # Conv: freeze all enc_* blocks.
                if args.model_type == "transformer":
                    frozen_params = (
                        list(model.encoder_transformer.parameters()) +
                        list(model.patch_to_embed.parameters()) +
                        list(model.enc_to_latent.parameters())
                    )
                else:
                    frozen_params = [
                        p for n, p in model.named_parameters()
                        if n.startswith("enc_")
                    ]

                for p in frozen_params:
                    p.requires_grad = False

                z_cycle = model.encode(audio_morph)                         # (B, L, D)

                for p in frozen_params:
                    p.requires_grad = True

                loss_cycle = F.mse_loss(z_cycle, z_lerp.detach())

                total_G_loss = (
                    total_G_loss
                    + args.morph_gan_weight * loss_morph_adv
                    + args.cycle_weight * loss_cycle
                )
                epoch_train_morph_adv += loss_morph_adv.item()
                epoch_train_cycle    += loss_cycle.item()

            # ---- Standalone cycle-consistency (encoder-training, decoupled) ----
            # E(D(z_mix)) ≈ z_mix with the decoder frozen: gradients flow
            # through the encoder only, pushing it to produce codes whose
            # cone-mixtures round-trip through the decoder.  Uses the universal
            # cone mixture (A·cosθ·Z_A + A·sinθ·Z_B) when --use_universal_mix is
            # active, else LERP (0.5·Z_A + 0.5·Z_B).  Unlike encoder_cone_loss
            # this is an *indirect* alignment — no MSE on the raw code geometry,
            # so there is no incentive to collapse the latent to a low-rank
            # subspace to trivially satisfy the homomorphism.
            if args.use_cycle and not args.use_morph_gan:
                audio_rolled_cyc = torch.roll(audio, shifts=1, dims=0)
                z_rolled_cyc = torch.roll(z, shifts=1, dims=0)

                if args.use_universal_mix:
                    # Same (A, θ) cone sampler as universal_mix_loss, annealed.
                    progress = (epoch - 1) / args.epochs if args.epochs > 0 else 1.0
                    g_min, g_max = cosine_gain_range(
                        args.universal_gain_min, args.universal_gain_max,
                        None if args.no_universal_gain_anneal else progress,
                    )
                    a_cyc = torch.ones(audio.shape[0], 1, 1, device=device)
                    if g_max > g_min:
                        use_gain = torch.rand(audio.shape[0], 1, 1, device=device) < args.universal_gain_prob
                        a_cyc = torch.where(
                            use_gain,
                            g_min + (g_max - g_min) * torch.rand(audio.shape[0], 1, 1, device=device),
                            a_cyc,
                        )
                    a_cyc = torch.where(a_cyc.abs() < 0.05, torch.zeros_like(a_cyc), a_cyc)
                    theta_cyc = (torch.pi / 2.0) * torch.rand(audio.shape[0], 1, 1, device=device)
                    is_mix_cyc = torch.rand(audio.shape[0], 1, 1, device=device) < args.universal_mix_fraction
                    theta_cyc = torch.where(is_mix_cyc, theta_cyc, torch.zeros_like(theta_cyc))
                    z_mix_cyc = a_cyc * (
                        torch.cos(theta_cyc) * z + torch.sin(theta_cyc) * z_rolled_cyc
                    )
                else:
                    # LERP fallback (matches the morph_gan cycle path)
                    z_mix_cyc = 0.5 * z + 0.5 * z_rolled_cyc

                # Decode the mixed latent (no grad to decoder — we train encoder)
                with torch.no_grad():
                    audio_morph_cyc = model.decode(
                        z_mix_cyc, original_length=audio.shape[-1],
                    )

                # Freeze the decoder so gradients flow through the encoder only
                if args.model_type == "transformer":
                    frozen_params_cyc = (
                        list(model.decoder_transformer.parameters()) +
                        list(model.latent_to_dec.parameters()) +
                        list(model.embed_to_patch.parameters())
                    )
                else:
                    frozen_params_cyc = [
                        p for n, p in model.named_parameters()
                        if n.startswith("dec_")
                    ]
                # Also freeze the soft-norm affine (scale/bias) — it is part of
                # the bottleneck, not the encoder's representational capacity.
                if model.soft_norm is not None:
                    frozen_params_cyc += list(model.soft_norm.parameters())

                for p in frozen_params_cyc:
                    p.requires_grad = False

                # Save soft-norm state: the re-encode of morphed audio would
                # otherwise pollute z_norm_clean (breaking the next batch's KL)
                # and update running_std with non-real-data statistics.
                _sn_clean_save = None
                _sn_running_std_save = None
                if model.soft_norm is not None:
                    _sn_clean_save = model.soft_norm.z_norm_clean
                    _sn_running_std_save = model.soft_norm.running_std.clone()

                z_cycle_reenc = model.encode(audio_morph_cyc)
                loss_cycle = F.mse_loss(z_cycle_reenc, z_mix_cyc.detach())

                # Restore soft-norm state
                if model.soft_norm is not None:
                    model.soft_norm.z_norm_clean = _sn_clean_save
                    model.soft_norm.running_std.copy_(_sn_running_std_save)

                for p in frozen_params_cyc:
                    p.requires_grad = True

                total_G_loss = total_G_loss + args.cycle_weight * loss_cycle
                epoch_train_cycle += loss_cycle.item()

            # ---- Backprop Generator ----
            optimizer_G.zero_grad()
            total_G_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_G.step()
            if args.scheduler == "onecycle":
                scheduler_G.step()

            # ---- Bookkeeping ----
            epoch_train_loss  += total_G_loss.item()
            epoch_train_mag   += mag_loss.item()
            epoch_train_phase += phase_loss_val.item() if args.use_phase_loss else 0.0
            n_train_batches   += 1

            postfix = {
                "G":   f"{total_G_loss.item():.4f}",
                "mag": f"{mag_loss.item():.4f}",
            }
            if args.use_gan or args.use_morph_gan:
                postfix["D"] = f"{loss_D.item():.4f}"
                postfix["adv"] = f"{loss_adv.item():.4f}"
            if args.use_latent_mixup:
                postfix["mixup"] = f"{loss_mixup.item():.4f}"
            if args.use_morph_gan:
                postfix["morph"] = f"{loss_morph_adv.item():.4f}"
                postfix["cycle"] = f"{loss_cycle.item():.4f}"
            if args.use_cycle and not args.use_morph_gan:
                postfix["cyc"] = f"{loss_cycle.item():.4f}"
            if args.use_soft_norm:
                postfix["KL"] = f"{loss_soft_norm_kl.item():.4f}"
            if args.use_implicit_linearity:
                postfix["impl"] = f"{loss_implicit.item():.4f}"
            if args.use_universal_mix:
                postfix["univ"] = f"{loss_univ.item():.4f}"
            if args.use_encoder_cone:
                postfix["enc"] = f"{loss_enc_cone.item():.4f}"
            pbar.set_postfix(postfix)

        avg_train_total   = epoch_train_loss   / n_train_batches
        avg_train_mag     = epoch_train_mag    / n_train_batches
        avg_train_phase   = epoch_train_phase  / n_train_batches
        avg_train_loss_D  = epoch_train_loss_D / n_train_batches if (args.use_gan or args.use_morph_gan) else 0.0
        avg_train_loss_G_adv = epoch_train_loss_G_adv / n_train_batches if args.use_gan else 0.0
        avg_train_mixup   = epoch_train_mixup  / n_train_batches if args.use_latent_mixup else 0.0
        avg_train_implicit = epoch_train_implicit / n_train_batches if args.use_implicit_linearity else 0.0
        avg_train_univ    = epoch_train_univ   / n_train_batches if args.use_universal_mix else 0.0
        avg_train_morph_adv = epoch_train_morph_adv / n_train_batches if args.use_morph_gan else 0.0
        avg_train_cycle   = epoch_train_cycle  / n_train_batches if (args.use_morph_gan or args.use_cycle) else 0.0
        avg_train_soft_norm_kl = epoch_train_soft_norm_kl / n_train_batches if args.use_soft_norm else 0.0
        avg_train_enc_cone = epoch_train_enc_cone / n_train_batches if args.use_encoder_cone else 0.0

        # ================================================================
        #  VALIDATION
        # ================================================================
        val_metrics = run_validation(
            model, val_loader, loss_fn,
            use_phase=args.use_phase_loss,
            device=device,
            use_latent_mixup=args.use_latent_mixup,
            mixup_weight=args.mixup_weight,
            use_implicit_linearity=args.use_implicit_linearity,
            implicit_linearity_weight=args.implicit_linearity_weight,
            implicit_gain_min=args.implicit_gain_min,
            implicit_gain_max=args.implicit_gain_max,
            implicit_gain_prob=args.implicit_gain_prob,
            implicit_mix_fraction=args.implicit_mix_fraction,
            no_implicit_gain_anneal=args.no_implicit_gain_anneal,
            implicit_progress=(epoch - 1) / args.epochs if args.epochs > 0 else 1.0,
            use_universal_mix=args.use_universal_mix,
            universal_mix_weight=args.universal_mix_weight,
            universal_gain_min=args.universal_gain_min,
            universal_gain_max=args.universal_gain_max,
            universal_gain_prob=args.universal_gain_prob,
            universal_mix_fraction=args.universal_mix_fraction,
            no_universal_gain_anneal=args.no_universal_gain_anneal,
            universal_progress=(epoch - 1) / args.epochs if args.epochs > 0 else 1.0,
            universal_sampler=args.universal_sampler,
            use_encoder_cone=args.use_encoder_cone,
            encoder_cone_weight=args.encoder_cone_weight,
            encoder_cone_mode=args.encoder_cone_mode,
            encoder_cone_anneal_fraction=args.encoder_cone_anneal_fraction,
            encoder_cone_progress=(epoch - 1) / args.epochs if args.epochs > 0 else 1.0,
        )

        # ---- Plateau scheduler: step once per epoch on the val metric ----
        if args.scheduler == "plateau":
            scheduler_G.step(val_metrics["val_total"])
            if args.use_gan or args.use_morph_gan:
                scheduler_D.step(val_metrics["val_total"])

        # ================================================================
        #  LOGGING
        # ================================================================

        phase_str = ""
        if args.use_phase_loss:
            phase_str = f"  phase={avg_train_phase:.6f}"
        val_phase_str = ""
        if args.use_phase_loss:
            val_phase_str = f"  phase={val_metrics['val_phase']:.6f}"
        gan_str = ""
        if args.use_gan or args.use_morph_gan:
            gan_str = f"  D={avg_train_loss_D:.4f}  G_adv={avg_train_loss_G_adv:.4f}"
        mixup_str = ""
        val_mixup_str = ""
        if args.use_latent_mixup:
            mixup_str = f"  mixup={avg_train_mixup:.6f}"
            val_mixup_str = f"  mixup={val_metrics['val_mixup']:.6f}"
        implicit_str = ""
        val_implicit_str = ""
        if args.use_implicit_linearity:
            implicit_str = f"  impl={avg_train_implicit:.6f}"
            val_implicit_str = f"  impl={val_metrics['val_implicit']:.6f}"
        univ_str = ""
        val_univ_str = ""
        if args.use_universal_mix:
            univ_str = f"  univ={avg_train_univ:.6f}"
            val_univ_str = f"  univ={val_metrics['val_univ']:.6f}"
        morph_str = ""
        if args.use_morph_gan:
            morph_str = f"  morph_adv={avg_train_morph_adv:.6f}  cycle={avg_train_cycle:.6f}"
        cycle_str = ""
        if args.use_cycle and not args.use_morph_gan:
            cycle_str = f"  cycle={avg_train_cycle:.6f}"
        soft_norm_str = ""
        if args.use_soft_norm:
            soft_norm_str = f"  KL={avg_train_soft_norm_kl:.6f}"
        enc_cone_str = ""
        val_enc_cone_str = ""
        enc_w_str = ""
        if args.use_encoder_cone:
            enc_cone_str = f"  enc={avg_train_enc_cone:.6f}"
            val_enc_cone_str = f"  enc={val_metrics['val_enc_cone']:.6f}"
            enc_w = _encoder_cone_weight_at(
                args.encoder_cone_weight,
                (epoch - 1) / args.epochs if args.epochs > 0 else 1.0,
                args.encoder_cone_anneal_fraction,
            )
            enc_w_str = f"  encλ={enc_w:.3f}"

        # Effective gain range this epoch for the annealed-gain losses
        # (progress is the same (epoch-1)/epochs value passed to the losses).
        gain_log: Dict[str, float] = {}
        gain_range_str = ""
        if args.use_implicit_linearity:
            g_lo, g_hi = cosine_gain_range(
                args.implicit_gain_min, args.implicit_gain_max,
                None if args.no_implicit_gain_anneal
                else ((epoch - 1) / args.epochs if args.epochs > 0 else 1.0),
            )
            gain_log["train/implicit_gain_min"] = g_lo
            gain_log["train/implicit_gain_max"] = g_hi
            gain_range_str += f"  impl-gain∈[{g_lo:.2f},{g_hi:.2f}]"
        if args.use_universal_mix:
            g_lo, g_hi = cosine_gain_range(
                args.universal_gain_min, args.universal_gain_max,
                None if args.no_universal_gain_anneal
                else ((epoch - 1) / args.epochs if args.epochs > 0 else 1.0),
            )
            gain_log["train/universal_gain_min"] = g_lo
            gain_log["train/universal_gain_max"] = g_hi
            gain_range_str += f"  univ-gain∈[{g_lo:.2f},{g_hi:.2f}]"

        print(
            f"[Epoch {epoch:4d}/{args.epochs}]  "
            f"train: total={avg_train_total:.6f}  mag={avg_train_mag:.6f}"
            f"{phase_str}{gan_str}{mixup_str}{implicit_str}{univ_str}{enc_cone_str}{enc_w_str}{morph_str}{cycle_str}{soft_norm_str}"
            f"{gain_range_str}"
            f"  |  val: total={val_metrics['val_total']:.6f}"
            f"  mag={val_metrics['val_mag']:.6f}"
            f"{val_mixup_str}{val_implicit_str}{val_univ_str}{val_enc_cone_str}{val_phase_str}"
        )

        current_lr = scheduler_G.get_last_lr()[0]

        wandb_log_dict = {
            "epoch":              epoch,
            "train/total_loss":   avg_train_total,
            "train/mag_loss":     avg_train_mag,
            "train/phase_loss":   avg_train_phase,
            "train/lr":           current_lr,
            "val/total_loss":     val_metrics["val_total"],
            "val/mag_loss":       val_metrics["val_mag"],
            "val/phase_loss":     val_metrics["val_phase"],
        }
        if args.use_gan or args.use_morph_gan:
            wandb_log_dict["train/loss_D"] = avg_train_loss_D
            wandb_log_dict["train/loss_G_adv"] = avg_train_loss_G_adv
        if args.use_latent_mixup:
            wandb_log_dict["train/mixup_loss"] = avg_train_mixup
            wandb_log_dict["val/mixup_loss"] = val_metrics["val_mixup"]
        if args.use_implicit_linearity:
            wandb_log_dict["train/implicit_linearity_loss"] = avg_train_implicit
            wandb_log_dict["val/implicit_linearity_loss"] = val_metrics["val_implicit"]
        if args.use_universal_mix:
            wandb_log_dict["train/universal_mix_loss"] = avg_train_univ
            wandb_log_dict["val/universal_mix_loss"] = val_metrics["val_univ"]
        if args.use_morph_gan:
            wandb_log_dict["train/morph_adv_loss"] = avg_train_morph_adv
            wandb_log_dict["train/cycle_loss"] = avg_train_cycle
        if args.use_cycle and not args.use_morph_gan:
            wandb_log_dict["train/cycle_loss"] = avg_train_cycle
        if args.use_soft_norm:
            wandb_log_dict["train/soft_norm_kl_loss"] = avg_train_soft_norm_kl
        if args.use_encoder_cone:
            wandb_log_dict["train/encoder_cone_loss"] = avg_train_enc_cone
            wandb_log_dict["val/encoder_cone_loss"] = val_metrics["val_enc_cone"]
            wandb_log_dict["train/encoder_cone_weight"] = enc_w

        wandb_log_dict.update(gain_log)

        wandb.log(wandb_log_dict, step=epoch)

        # ================================================================
        #  EARLY STOPPING & BEST-MODEL CHECKPOINT
        # ================================================================
        is_best = early_stopper.step(val_metrics["val_total"])

        if is_best:
            save_best_checkpoint(
                model, args.run_name,
                optimizer_G, scheduler_G, epoch,
                discriminator, optimizer_D, scheduler_D,
                model_config=model_config or None,
            )

        if early_stopper.early_stop:
            print(
                f"[EarlyStop] No improvement for {args.patience} epochs -- "
                f"stopping at epoch {epoch}."
            )
            break

        # ================================================================
        #  TOPOLOGICAL EVALUATION HOOK  (every 25 epochs)
        # ================================================================
        if not args.fast_eval and epoch % TOPO_INTERVAL == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                topo_metrics = _run_topological_eval(
                    model, topo_audio_a_batch, topo_audio_b_batch,
                )
            model.train()

            # ---- Save audio samples for the first pair ----
            for tag, tensor in [
                ("rec_A", topo_metrics["Decoded_A"]),
                ("rec_B", topo_metrics["Decoded_B"]),
                ("interp_mid", topo_metrics["Decoded_Mix"]),
            ]:
                waveform = tensor[0].detach().cpu()          # (1, T)
                out_path = topo_samples_dir / f"epoch_{epoch:04d}_{tag}.wav"
                torchaudio.save(str(out_path), waveform, args.sample_rate)
            print(f"[AudioSamples] Saved to {topo_samples_dir}")

            wandb.log(
                {
                    "eval/decoder_additivity_cosine_sim": topo_metrics[
                        "decoder_additivity_cosine_sim"
                    ],
                    "eval/encoder_additivity_mse": topo_metrics[
                        "encoder_additivity_mse"
                    ],
                    "eval/latent_density": topo_metrics[
                        "latent_density"
                    ],
                    "eval/latent_convexity": topo_metrics[
                        "latent_convexity"
                    ],
                },
                step=epoch,
            )
            print(
                f"[TopoEval epoch {epoch:4d}]  "
                f"dec_cosine={topo_metrics['decoder_additivity_cosine_sim']:.6f}  "
                f"enc_mse={topo_metrics['encoder_additivity_mse']:.6f}  "
                f"lat_density={topo_metrics['latent_density']:.6f}  "
                f"lat_convexity={topo_metrics['latent_convexity']:.6f}"
            )

    # ---- Save final checkpoint (always) ----
    save_checkpoint(
        model, args.run_name,
        optimizer_G, scheduler_G, epoch,
        discriminator, optimizer_D, scheduler_D,
        model_config=model_config or None,
    )

    # ---- Report best ----
    if early_stopper.best_score is not None:
        print(f"[Best] Best validation loss: {early_stopper.best_score:.6f}")

    # ---- Cleanup ----
    wandb.finish()
    print("[Done] Training complete.")


# =============================================================================
#  CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toy 1D Conv AE -- Phase Regularization Ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--data_dir", "--data_dirs", dest="data_dirs", type=str, nargs="+",
        required=True,
        help="One or more dataset directories to aggregate "
             "(e.g. /data/wav48_silence_trimmed "
             "/data/fsd50k/FSD50K.dev_audio "
             "/data/fma_small).",
    )
    parser.add_argument(
        "--preprocess_cache_dir", type=str,
        default=DEFAULT_PREPROCESSED_CACHE_DIR,
        help="Base directory for the one-time preprocessed audio cache. "
             "A per-sample-rate subdirectory 'sr<rate>' is created inside it. "
             "If the matching manifest.json already exists, preprocessing is skipped.",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="Number of training epochs (default: 1000).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--limit_train_batches", type=int, default=0,
        help="If >0, cap the number of training batches per epoch (0 = full "
             "pass). Useful for fast hyperparameter iteration: the OneCycle "
             "scheduler's steps_per_epoch is clamped to the same value so the "
             "LR schedule still anneals correctly over --epochs mini-epochs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device string, e.g. 'cuda:0', 'cuda:1', 'cpu' (default: 'cuda:0')."
             " Falls back to CPU if the requested device is unavailable.",
    )

    # Audio
    parser.add_argument(
        "--chunk_seconds", type=float, default=2.0,
        help="Duration of audio chunks in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate in Hz for preprocessing and training "
             "(e.g. 16000, 24000, 44100, 48000; default: 16000).",
    )

    # Model architecture
    parser.add_argument(
        "--model_type", type=str, default="transformer",
        choices=["conv", "transformer"],
        help="Model architecture: 'conv' (CNN) or 'transformer' (SAME-inspired). "
             "Default: 'conv'.",
    )

    # Phase experiment
    parser.add_argument(
        "--use_phase_loss", action="store_true", default=False,
        help="If set, add IF/GD/Complex-Distance phase terms to the SAME loss.",
    )

    # Fast evaluation
    parser.add_argument(
        "--fast_eval", action="store_true", default=True,
        help="If set, skip expensive topological metrics (latent_convexity, "
             "latent_density) and audio waveform saving during training.",
    )

    # Adversarial training (GAN)
    parser.add_argument(
        "--use_gan", dest="use_gan", action="store_true", default=True,
        help="Train an STFT-domain PatchGAN discriminator alongside "
             "the autoencoder using the LSGAN objective "
             "(default: True — best from hparam search).",
    )
    parser.add_argument(
        "--no_use_gan", dest="use_gan", action="store_false",
        help="Disable adversarial training.",
    )
    parser.add_argument(
        "--adv_loss_weight", type=float, default=0.05,
        help="Weight λ for the adversarial loss in the generator objective: "
             "loss_G = loss_SAME + λ * loss_adv  (default: 0.05 — best from "
             "hparam search).",
    )

    # Activation function
    parser.add_argument(
        "--activation", type=str, default="leaky_relu",
        choices=["leaky_relu", "silu", "snake"],
        help="Base activation for the Encoder and initial Decoder layers "
             "of the Transformer: 'leaky_relu' (default) or 'silu'.  The CNN "
             "baseline uses 'leaky_relu' (or 'snake').  For the Transformer, "
             "periodic Snake is applied via --decoder_num_snake_layers; "
             "legacy 'snake' maps to base 'leaky_relu' + Snake on all decoder "
             "layers.",
    )
    parser.add_argument(
        "--decoder_num_snake_layers", type=int, default=2,
        help="Number of final decoder layers to equip with periodic Snake "
             "activation per SAME paper (default: 2).  The first "
             "num_layers - K decoder layers use the base activation.",
    )

    # Bottleneck / compression ablation knobs (Transformer model only)
    # ------------------------------------------------------------------
    # Per-scalar compression = patch_size / latent_dim, e.g. 256/64 = 4:1.
    #   * "code compression": shrink latent_dim at fixed patch_size (more
    #     aggressive per-scalar rate at the same temporal resolution);
    #   * "time compression": raise patch_size (= temporal stride Dt) at
    #     fixed latent_dim (fewer frames per second).
    # d_model is exposed so the transformer width can be adapted to very
    # small latent_dim without starving the bottleneck projection.
    parser.add_argument(
        "--latent_dim", type=int, default=64,
        help="Bottleneck channel width d (default: 64).  Lower = higher "
             "per-scalar 'code compression' at fixed patch_size.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256,
        help="Waveform patch size = temporal downsampling stride Dt "
             "(default: 256).  Raise for 'time compression' (fewer latent "
             "frames per second), e.g. 1024/2048.",
    )
    parser.add_argument(
        "--d_model", type=int, default=512,
        help="Transformer model width (default: 512; must be divisible by "
             "nhead=8).",
    )

    # Final LayerNorm control (Transformer model only)
    parser.add_argument(
        "--encoder_final_norm", dest="encoder_final_norm",
        action="store_true", default=None,
        help="Apply a final LayerNorm after the encoder transformer stack "
             "(transformer model only).  When unspecified (default: None), "
             "follows the auto behaviour: OFF while latent mixup needs a "
             "scale-sensitive bottleneck, otherwise ON.",
    )
    parser.add_argument(
        "--no_encoder_final_norm", dest="encoder_final_norm",
        action="store_false",
        help="Disable the final encoder LayerNorm (scale-sensitive encoder).",
    )
    parser.add_argument(
        "--decoder_final_norm", dest="decoder_final_norm",
        action="store_true", default=None,
        help="Apply a final LayerNorm after the decoder transformer stack "
             "(transformer model only).  When unspecified (default: None), "
             "follows the auto behaviour: OFF while latent mixup needs a "
             "scale-sensitive bottleneck, otherwise ON.",
    )
    parser.add_argument(
        "--no_decoder_final_norm", dest="decoder_final_norm",
        action="store_false",
        help="Disable the final decoder LayerNorm (scale-sensitive decoder).",
    )

    # Regularisation
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout probability in encoder/decoder (default: 0.0). "
             "Set to 0.0 to disable.",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.10,
        help="Fraction of tracks for validation (default: 0.10).",
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience in epochs (default: 15 — best from "
             "hparam search).",
    )
    parser.add_argument(
        "--min_delta", type=float, default=1e-5,
        help="Minimum val loss improvement to count as progress (default: 1e-5 — "
             "best from hparam search).",
    )
    parser.add_argument(
        "--scheduler", type=str, default="plateau",
        choices=["onecycle", "plateau"],
        help="LR scheduler. 'onecycle': OneCycleLR with 5%% warmup, anneals the "
             "LR to ~0 over exactly --epochs (fixed-horizon schedule). "
             "'plateau': ReduceLROnPlateau (factor 0.5, patience 5, min_lr "
             "1e-6) that decays the LR whenever the validation loss plateaus — "
             "coherent with early stopping: the LR is already low/converged by "
             "the time early stopping fires, so set a large --epochs cap and "
             "let the run stop on its own (default: plateau).",
    )

    # Resume / Recovery
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--wandb_run_id", type=str, default=None,
        help="WandB run ID of the interrupted session to resume.",
    )

    # Latent Mixup (Trigonometric, in-batch rolling)
    parser.add_argument(
        "--use_latent_mixup", dest="use_latent_mixup", action="store_true", default=True,
        help="Add a trigonometric constant-power latent mixup loss that forces "
             "the decoder to map a latent trig-interpolation directly to the "
             "corresponding audio-domain trig-mix.  Uses an in-batch rolling "
             "trick to avoid a 2× compute cost (default: True — best from "
             "hparam search).",
    )
    parser.add_argument(
        "--no_use_latent_mixup", dest="use_latent_mixup", action="store_false",
        help="Disable the latent mixup (additivity) loss.",
    )
    parser.add_argument(
        "--mixup_weight", type=float, default=1.0,
        help="Weight λ for the latent mixup loss: "
             "total_G_loss = loss_base + λ * loss_mixup  (default: 1.0).",
    )

    # Torres-style implicit linearity (implicit homogeneity + additivity)
    # Torres, Moussallam & Meseguer-Brocal, "Learning Linearity in Audio
    # Consistency Autoencoders via Implicit Regularization", ICASSP 2026,
    # arXiv:2510.23530 — adapted from their consistency-model recipe to the
    # plain AE: decode a·z (gain-scaled latent, target a·x) for homogeneity and
    # decode z_u + z_v (sum of individual latents, target u+v) for additivity.
    parser.add_argument(
        "--use_implicit_linearity", action="store_true", default=False,
        help="Add the Torres-style implicit linearity regularizer "
             "(implicit homogeneity + additivity, see losses.py).  Used for the "
             "matched ablation against --use_latent_mixup: same architecture, "
             "data and loss, different linearity-inducing mechanism.",
    )
    parser.add_argument(
        "--implicit_linearity_weight", type=float, default=1.0,
        help="Weight λ for the implicit linearity loss: "
             "total_G_loss = loss_base + λ * loss_implicit  (default: 1.0).",
    )
    parser.add_argument(
        "--implicit_gain_min", type=float, default=0.0,
        help="Lower bound of the homogeneity gain range U(gain_min, gain_max) "
             "(paper: 0; annealed to 1 over training).",
    )
    parser.add_argument(
        "--implicit_gain_max", type=float, default=3.0,
        help="Upper bound of the homogeneity gain range U(gain_min, gain_max) "
             "(paper: 3; annealed to 1 over training).",
    )
    parser.add_argument(
        "--implicit_gain_prob", type=float, default=0.8,
        help="Probability of applying a random gain to each sample (paper: 0.8).",
    )
    parser.add_argument(
        "--implicit_mix_fraction", type=float, default=0.5,
        help="Fraction of each batch turned into artificial mixtures "
             "(paper: batch size doubles with 50%% mixtures).",
    )
    parser.add_argument(
        "--no_implicit_gain_anneal", action="store_true",
        help="Disable the cosine anneal of the gain range toward (1, 1); "
             "the fixed (gain_min, gain_max) range is used for the whole run.",
    )

    # Universal positive-weight interpolation (constant-power cone)
    # Supervises the decoder on the ENTIRE positive-weight mixing cone via the
    # polar parametrisation (w1, w2) = A·(cosθ, sinθ):
    #   D(A·cosθ·Z_A + A·sinθ·Z_B) ≈ A·cosθ·X_A + A·sinθ·X_B
    # with θ ~ U(0, π/2) and A ~ U(gain_min, gain_max).  Subsumes the trig
    # latent-mixup loss (A ≡ 1) and the implicit homogeneity objective (θ = 0),
    # so no interpolation law (lerp / log / parabolic / exponential / …) is
    # out-of-distribution.  See toy_training/losses.py::universal_mix_loss.
    parser.add_argument(
        "--use_universal_mix", action="store_true", default=False,
        help="Add the universal interpolation (constant-power cone) loss: "
             "train the decoder to interpolate EVERY positive-weight path "
             "accurately instead of memorising one curve.",
    )
    parser.add_argument(
        "--universal_mix_weight", type=float, default=1.0,
        help="Weight λ for the universal interpolation loss: "
             "total_G_loss = loss_base + λ * loss_univ  (default: 1.0).",
    )
    parser.add_argument(
        "--universal_gain_min", type=float, default=0.0,
        help="Lower bound of the polar radius (energy scale) "
             "A ~ U(gain_min, gain_max) (default: 0.0).",
    )
    parser.add_argument(
        "--universal_gain_max", type=float, default=3.0,
        help="Upper bound of the polar radius (energy scale) A (default: 3.0).",
    )
    parser.add_argument(
        "--universal_gain_prob", type=float, default=0.8,
        help="Probability of sampling a random energy scale A per sample "
             "(else A = 1, i.e. the plain trigonometric mixup; default: 0.8).",
    )
    parser.add_argument(
        "--universal_mix_fraction", type=float, default=0.5,
        help="Fraction of each batch turned into (A, θ) interpolations; the "
             "remainder uses θ = 0 (gain-scaled reconstruction), keeping the "
             "reconstruction objective explicit (default: 0.5).",
    )
    parser.add_argument(
        "--no_universal_gain_anneal", action="store_true",
        help="Disable the cosine anneal of the polar radius toward (1, 1); "
             "the fixed (universal_gain_min, universal_gain_max) range is used "
             "for the whole run.",
    )
    parser.add_argument(
        "--universal_sampler", type=str, default="polar",
        choices=["polar", "cart_square", "cart_iso"],
        help="Cone-density sampler for the universal loss "
             "(polar-vs-cartesian sampler ablation): 'polar' "
             "(default, principled: isotropic-in-angle, decoupled radial "
             "curriculum), 'cart_square' (naive cartesian U(0,S)^2, "
             "anisotropic, expected worse), 'cart_iso' (cartesian weights "
             "induced by the polar law — control, expected ≈ polar).",
    )

    # Encoder-side cone consistency (encoder homomorphism)
    # MSE(E(w1·A + w2·B), w1·E(A) + w2·E(B)) on clean soft-norm codes — the
    # encoder mirror of the trig-mixup / universal decoder losses.  Sampler
    # follows the active decoder loss unless overridden with a mode.
    parser.add_argument(
        "--use_encoder_cone", action="store_true", default=False,
        help="Add the encoder-side cone-consistency (homomorphism) loss "
             "MSE(E(w1·A + w2·B), w1·E(A) + w2·E(B)) on clean soft-norm codes, "
             "with weights sampled like the active decoder loss "
             "(universal cone if --use_universal_mix, else the constant-power "
             "trig arc). Requires --use_soft_norm.",
    )
    parser.add_argument(
        "--encoder_cone_weight", type=float, default=1.0,
        help="Weight λ for the encoder cone loss: "
             "total_G_loss = loss_base + λ * loss_enc_cone (default: 1.0).",
    )
    parser.add_argument(
        "--encoder_cone_mode", type=str, default="auto",
        choices=["auto", "trig", "universal"],
        help="Sampler for the encoder cone term: 'auto' uses the universal "
             "cone when --use_universal_mix is active and the constant-power "
             "trig arc otherwise (default); 'trig' forces the constant-power "
             "arc; 'universal' forces the full (A, θ) cone sampler "
             "(with the --universal_* gain parameters).",
    )
    parser.add_argument(
        "--encoder_cone_anneal_fraction", type=float, default=0.0,
        help="Fraction of the run over which the encoder-cone weight λ is "
             "cosine-annealed from its full value to 0 (default 0.0 = no "
             "anneal). E.g. 0.25 keeps λ constant for the first 75%% of the "
             "run and anneals it to 0 over the last 25%%, letting the "
             "soft-norm KL / reconstruction re-spread the latent codes at the "
             "end so the encoder-homomorphism pressure does not leave a "
             "collapsed (low-rank, correlated) latent for a downstream "
             "diffusion model.",
    )

    # Morph GAN (LERP feature morphing + Cycle Consistency)
    parser.add_argument(
        "--use_morph_gan", action="store_true", default=False,
        help="If set, enable the LERP feature-morphing dual behaviour: "
             "the discriminator learns that audio mixes are fake, and a "
             "cycle-consistency loss ensures that morphing latents decode "
             "to audio that re-encodes back to the same latent.",
    )
    parser.add_argument(
        "--morph_gan_weight", type=float, default=0.1,
        help="Weight λ for the generator's morphing adversarial loss: "
             "loss_G += λ * loss_morph_adv  (default: 0.1).",
    )
    parser.add_argument(
        "--cycle_weight", type=float, default=0.0,
        help="Weight for the Cycle Consistency loss: "
             "loss_G += λ * loss_cycle  (default: 0.0).",
    )

    # Standalone cycle-consistency (decoupled from morph_gan)
    # Trains the ENCODER (not the decoder) to be consistent with the decoder's
    # cone behavior: E(D(z_mix)) ≈ z_mix.  When --use_universal_mix is active the
    # mixture uses the same (A, θ) cone sampler; otherwise it uses LERP.
    # Unlike encoder_cone_loss (a direct MSE on the raw code geometry that can
    # collapse the latent), this is an indirect pressure: the encoder can use
    # a rich, high-dimensional code as long as the round-trip is consistent —
    # there is no incentive to reduce rank.
    parser.add_argument(
        "--use_cycle", action="store_true", default=False,
        help="Add a standalone cycle-consistency loss E(D(z_mix)) ≈ z_mix that "
             "trains the encoder (decoder frozen) to be consistent with the "
             "decoder's cone behavior.  Decoupled from --use_morph_gan: no "
             "adversarial morph negatives, just the round-trip MSE.  Uses the "
             "universal cone mixture (A·cosθ·Z_A + A·sinθ·Z_B) when "
             "--use_universal_mix is active, else LERP.  See --cycle_weight.",
    )

    # SAME Soft-Norm Bottleneck
    parser.add_argument(
        "--use_soft_norm", dest="use_soft_norm", action="store_true", default=True,
        help="Enable the SAME Soft-Normalisation Bottleneck (EMA std tracking, "
             "per-channel affine, Gaussian noise injection) and the dual-axis "
             "KL regularisation loss (Eq. 1) (default: True — best from hparam "
             "search).",
    )
    parser.add_argument(
        "--no_use_soft_norm", dest="use_soft_norm", action="store_false",
        help="Disable the soft-norm bottleneck and KL loss.",
    )
    parser.add_argument(
        "--soft_norm_kl_weight", type=float, default=1.0,
        help="Weight λ for the soft-norm KL regularisation loss: "
             "total_G_loss += λ * loss_soft_norm_kl  (default: 1.0).",
    )

    # Logging
    parser.add_argument(
        "--run_name", type=str, default="toy_ae",
        help="Name for the WandB run and checkpoint file.",
    )

    return parser.parse_args()


# =============================================================================
#  Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        print(
            "ERROR: torchaudio is required. Install with:\n"
            "    pip install torchaudio",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()

    print("=" * 60)
    print("  Toy AE -- Phase Regularization Ablation")
    print("=" * 60)
    for key, val in sorted(vars(args).items()):
        print(f"  {key:20s}: {val}")
    print("=" * 60)

    train(args)
