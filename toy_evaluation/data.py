#!/usr/bin/env python3
"""
Shared multi-dataset evaluation data layer
============================================

Every evaluation script in ``toy_evaluation/`` uses this module to obtain the
validation set and load audio, so that all protocols evaluate **exactly the
held-out data training used** — same preprocessing, same track-level split,
same chunking.

Data pipeline (mirrors ``toy_training/train_toy.py``):

1. ``preprocess_dataset_cache`` (training) walks one or more dataset roots,
   collects ``.wav`` / ``.flac`` / ``.mp3`` files, and caches each as a mono,
   digital-silence-trimmed, peak-normalised (0.95 headroom), resampled-to-16kHz
   float32 tensor under ``<cache_dir>/sr<rate>/``, with a ``manifest.json``
   recording ``tracks: {track_id: [pt_path, ...]}``.  Track IDs are
   domain-namespaced: ``"<dataset_root_basename>::<track>"`` where *track* is
   the file stem for flat layouts and the parent directory name for
   hierarchical layouts.

2. ``_build_datasets`` (training) sorts the track IDs, shuffles them with
   ``random.Random(split_seed=42)`` and reserves the first ``val_frac=0.10``
   fraction as validation tracks.

3. This module replicates (1)–(2) and serves the validation stems
   ``(pt_path, stem_name, track_id)`` plus a chunk extractor identical to
   ``AudioPreprocessedDataset`` (centre chunk for evaluation).

If the cache does not exist yet, it is built on demand by importing the
training preprocessing routine (requires the training dependencies).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# =============================================================================
#  Constants (mirror toy_training/train_toy.py)
# =============================================================================

MODEL_SAMPLE_RATE = 16000   # models are trained at 16 kHz
CLIP_DURATION = 2.0         # seconds (geometry battery; 10s used for recon quality)
VAL_FRAC = 0.10             # fraction of tracks reserved for validation
SPLIT_SEED = 42             # deterministic track shuffle seed

DEFAULT_DATA_DIRS = [
    "/data/wav48_silence_trimmed",
    "/data/fsd50k/FSD50K.dev_audio",
    "/data/fma_small",
]
DEFAULT_CACHE_DIR = "/data/preprocessed_cache"


# =============================================================================
#  Cache manifest
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


def get_cache_manifest(
    data_dirs,
    cache_dir: Optional[str] = None,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> Dict:
    """Load the preprocessing manifest for *sample_rate*; build it if missing.

    Returns the manifest dict with keys ``sample_rate``, ``num_files``,
    ``num_cached``, ``tracks`` (``{track_id: [pt_path, ...]}``) and
    ``domains`` (``{track_id: domain}``) — exactly the format produced by
    ``toy_training.train_toy.preprocess_dataset_cache``.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_path = Path(cache_dir) / f"sr{sample_rate}"
    manifest_file = cache_path / "manifest.json"

    if manifest_file.exists():
        with open(manifest_file, "r") as fh:
            manifest = json.load(fh)
        if manifest.get("sample_rate") == sample_rate:
            return manifest
        print(f"[Data] Cache at {manifest_file} has sr="
              f"{manifest.get('sample_rate')} but target is sr={sample_rate} "
              f"— rebuilding.")

    # ---- Build the cache exactly like training does ----
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from toy_training.train_toy import preprocess_dataset_cache
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Cache manifest missing at {manifest_file} and the training "
            f"preprocessing routine could not be imported ({exc}). "
            f"Run toy_training/train_toy.py once (or build the cache manually)."
        ) from exc
    dirs = _normalise_data_dirs(data_dirs)
    return preprocess_dataset_cache(dirs, str(cache_path), target_sr=sample_rate)


# =============================================================================
#  Train / validation split (track-level, mirrors _build_datasets)
# =============================================================================

def get_validation_tracks(
    manifest: Dict,
    val_frac: float = VAL_FRAC,
    split_seed: int = SPLIT_SEED,
) -> Set[str]:
    """Return the set of domain-namespaced track IDs reserved for validation.

    Replicates ``toy_training.train_toy._build_datasets`` exactly: sort the
    track IDs, shuffle deterministically with ``random.Random(split_seed)``,
    reserve the first ``int(n_tracks * val_frac)`` (min 1) tracks.
    """
    tracks = sorted(manifest.get("tracks", {}).keys())
    if len(tracks) < 2:
        raise RuntimeError(
            f"Need at least 2 tracks for a train/val split; got {len(tracks)}."
        )
    rng = random.Random(split_seed)
    rng.shuffle(tracks)
    n_val = max(1, int(len(tracks) * val_frac))
    return set(tracks[:n_val])


def collect_validation_stems(
    data_dirs=None,
    cache_dir: Optional[str] = None,
    val_frac: float = VAL_FRAC,
    split_seed: int = SPLIT_SEED,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> List[Tuple[str, str, str]]:
    """Return ``(pt_path, stem_name, track_id)`` for every validation stem.

    *data_dirs* is only used to build the cache if it does not exist yet; the
    actual stem list comes from the manifest, so the evaluation set is always
    exactly the set training used for validation (over all datasets).

    Returns a deterministically sorted list (sorted by path), which makes
    pair generation reproducible across checkpoints.
    """
    if data_dirs is None:
        data_dirs = DEFAULT_DATA_DIRS
    manifest = get_cache_manifest(data_dirs, cache_dir, sample_rate=sample_rate)
    val_tracks = get_validation_tracks(manifest, val_frac=val_frac, split_seed=split_seed)

    stems: List[Tuple[str, str, str]] = []
    for track_id, paths in manifest["tracks"].items():
        if track_id not in val_tracks:
            continue
        for p in paths:
            stem_name = os.path.splitext(os.path.basename(p))[0]
            stems.append((p, stem_name, track_id))
    stems.sort()
    return stems


# =============================================================================
#  Audio loading (mirrors AudioPreprocessedDataset)
# =============================================================================

def load_stem_clip(
    pt_path: str,
    clip_samples: int,
    centre: bool = True,
    seed: Optional[int] = None,
):
    """Load a cached ``.pt`` waveform and extract a chunk.

    Mirrors ``AudioPreprocessedDataset.__getitem__``: centre chunk when
    *centre* is True (evaluation), random chunk otherwise; symmetric
    zero-padding for clips shorter than *clip_samples*.

    Returns a float32 tensor of shape ``(1, clip_samples)``.
    """
    import torch

    waveform = torch.load(pt_path, map_location="cpu", weights_only=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    total_len = waveform.shape[-1]
    if total_len < clip_samples:
        pad_total = clip_samples - total_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
    else:
        if centre:
            start = (total_len - clip_samples) // 2
        else:
            if seed is not None:
                rng = random.Random(seed)
                start = rng.randint(0, total_len - clip_samples)
            else:
                start = random.randint(0, total_len - clip_samples)
        waveform = waveform[:, start:start + clip_samples]

    return waveform.float()  # (1, clip_samples)


def stem_to_numpy(
    pt_path: str,
    clip_samples: int,
    centre: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Load a cached stem as a mono float32 numpy array of shape ``(T,)``."""
    wav = load_stem_clip(pt_path, clip_samples, centre=centre, seed=seed)
    return wav.squeeze(0).numpy().astype(np.float32)


# =============================================================================
#  Pair generation (unified across all protocols)
# =============================================================================

def generate_pairs(
    stems: List[Tuple[str, str, str]],
    num_pairs: int,
    seed: int = 42,
    mode: str = "random",
) -> List[Tuple[int, int]]:
    """Return *num_pairs* index-pairs from the stem list (no self-pairs).

    Parameters
    ----------
    mode : 'random' | 'intra' | 'inter'
        - ``random`` — any two different stems (default; with the unified
          multi-dataset pool this is the only mode that reliably produces
          enough pairs, since stems are file-unique).
        - ``intra``  — both stems must share the same stem name.
        - ``inter``  — stems must have different stem names.
    """
    rng = random.Random(seed)
    n = len(stems)
    if n < 2:
        raise ValueError(f"Need at least 2 stems, got {n}")

    def _compatible(idx_a: int, idx_b: int) -> bool:
        if idx_a == idx_b:
            return False
        sa, sb = stems[idx_a][1], stems[idx_b][1]
        if mode == "random":
            return True
        if mode == "intra":
            return sa == sb
        if mode == "inter":
            return sa != sb
        return True

    pairs: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = num_pairs * 200
    while len(pairs) < num_pairs and attempts < max_attempts:
        i, j = rng.sample(range(n), 2)
        if not _compatible(i, j):
            attempts += 1
            continue
        if (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))
        attempts += 1
    if len(pairs) < num_pairs:
        print(f"  ⚠ Only {len(pairs)} pairs generated "
              f"(mode={mode}, requested={num_pairs}).")
    return pairs


# =============================================================================
#  Legacy VCTK-only split (kept for backward compatibility)
# =============================================================================

def get_validation_track_set(
    data_dir: str,
    val_frac: float = VAL_FRAC,
    split_seed: int = SPLIT_SEED,
) -> Set[str]:
    """Legacy VCTK-only helper: return the set of track names reserved for
    validation.

    This mirrors the *old* (pre-unified-dataset) training split and is kept
    only so old callers do not break.  New code should use
    :func:`collect_validation_stems` / :func:`get_validation_tracks` instead.
    """
    tracks: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(data_dir):
        for fname in filenames:
            if not fname.endswith("_mic2.flac"):
                continue
            if dirpath == data_dir:
                track = fname.replace("_mic2.flac", "")
            else:
                track = os.path.basename(dirpath)
            if track not in tracks:
                tracks.append(track)
    if len(tracks) < 2:
        raise RuntimeError(
            f"Need at least 2 tracks for a train/val split; got {len(tracks)} "
            f"under {data_dir}"
        )
    tracks.sort()
    rng = random.Random(split_seed)
    rng.shuffle(tracks)
    n_val = max(1, int(len(tracks) * val_frac))
    return set(tracks[:n_val])
