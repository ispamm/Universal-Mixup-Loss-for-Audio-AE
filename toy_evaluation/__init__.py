# toy_evaluation package

"""Shared utilities for the toy autoencoder evaluation suite.

The evaluation suite is meant to evaluate **exactly what training saw**.  To
that end every script in ``toy_evaluation/`` shares one data layer
(``toy_evaluation.data``) that:

1. loads the preprocessed multi-dataset cache manifest built by
   ``toy_training.train_toy.preprocess_dataset_cache`` (mono, digital-silence
   trimmed, peak-normalised to 0.95 headroom, resampled to 16 kHz, cached as
   ``.pt`` tensors);
2. replicates the **track-level 90/10 train/validation split** used by
   ``train_toy.py`` exactly (sorted track IDs, ``random.Random(42)`` shuffle,
   first 10 % reserved for validation);
3. exposes validation stems as ``(pt_path, stem_name, track_id)`` tuples and a
   chunk extractor that mirrors ``AudioPreprocessedDataset`` (centre chunk for
   evaluation).

This guarantees the held-out set is *identical* to the validation set training
used, over all three ingested datasets (VCTK, FSD50K, FMA-small), with no
train/val leakage and no preprocessing drift between training and evaluation.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Set, Tuple

# Backwards-compatible re-exports (old VCTK-only helpers).
from toy_evaluation.data import (
    CLIP_DURATION,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIRS,
    MODEL_SAMPLE_RATE,
    SPLIT_SEED,
    VAL_FRAC,
    collect_validation_stems,
    generate_pairs,
    get_cache_manifest,
    get_validation_track_set,
    get_validation_tracks,
    load_stem_clip,
    stem_to_numpy,
)

__all__ = [
    "CLIP_DURATION",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIRS",
    "MODEL_SAMPLE_RATE",
    "SPLIT_SEED",
    "VAL_FRAC",
    "collect_validation_stems",
    "generate_pairs",
    "get_cache_manifest",
    "get_validation_track_set",
    "get_validation_tracks",
    "load_stem_clip",
    "stem_to_numpy",
]
