#!/usr/bin/env bash
# =============================================================================
# 3-way sampler ablation — polar vs cartesian-square vs cartesian-iso
#
# Purpose: justify the polar parametrisation of the universal mixup loss by
# showing that (i) the "natural" cartesian default (uniform on the square,
# `cart_square`) is WORSE than `polar`, and (ii) a cartesian parametrisation
# induced to match the polar density (`cart_iso`) is EQUAL to `polar` —
# proving the polar choice is principled, not cosmetic.
#
# DESIGN: DECODER-ONLY ablation.  Only --universal_sampler differs across arms;
# everything else is identical.  To keep the comparison clean we keep OFF all
# auxiliary losses whose sampler is a SEPARATE hardcoded polar cone
# (independent of --universal_sampler), because mixing them in would muddy the
# decoder-only signal we are trying to isolate:
#   * --use_encoder_cone OFF (its default): encoder_cone_loss uses its own
#     polar sampler (encoder_cone_mode=auto->universal), untouched by
#     --universal_sampler.
#   * --use_cycle OFF (its default): the cycle path has a hardcoded polar cone
#     (train_toy.py:1521-1542), not wired to --universal_sampler.
#   * --use_morph_gan OFF (its default): same.
# The "encoder+decoder cone + cycle anti-collapse" recipe you are currently
# developing is a DIFFERENT, harder experiment and must be validated on its
# own; do not conflate it with this decoder-only sampler ablation.
#
# REUSING the polar arm: the existing checkpoint
#   toy_training/checkpoints/transformer_univ_full_best.pt
# (seed 42, universal_mix ON, NO encoder_cone, NO cycle -- the exact recipe
# below, minus --universal_sampler which defaults to "polar") IS the polar
# arm.  We therefore train ONLY the two cartesian arms and reuse that
# checkpoint for the polar reference.  Saves one training.
#
# Recipe (matches transformer_univ_full exactly, from its wandb config
# run-20260901_145853-l13bz58e -- verified command-line + config):
#   universal_mix (cone) loss ON, latent_mixup OFF, encoder_cone OFF, cycle OFF,
#   soft-norm KL ON (default True, lambda_KL=1), GAN ON (lambda_adv 0.05),
#   LeakyReLU + 2 final Snake decoder layers,
#   scale-sensitive bottleneck (final LayerNorm OFF, auto via universal_mix),
#   AdamW lr=1e-4 wd=1e-4, plateau scheduler, patience 15, min_delta 1e-5,
#   batch 64, 2 s chunks @ 16 kHz, latent_dim 64, patch 256 (defaults),
#   seed 42.
#   (transformer_univ_full was launched with --epochs 10000 + early stopping;
#    we use --epochs 300 + the same early stopping for the two new arms so
#    they stop at comparable val-plateau. The polar reference already
#    early-stopped at ~ep 218 on the same val split.)
#
# Expected outcome (per docs §8.4):
#   polar(=transformer_univ_full) ~= cart_iso   (same backprop at equal dist, §8.1)
#   cart_square < polar                          (anisotropic in theta, §8.2)
#
# PARALLEL on 2 GPUs: cart_square on GPU A, cart_iso on GPU B.  Both launched
# in background; the script waits for both to finish.
# =============================================================================
set -euo pipefail

# ---- EDIT THESE to your two free GPUs ---------------------------------------
DEVICE_A="${DEVICE_A:-cuda:0}"   # for cart_square
DEVICE_B="${DEVICE_B:-cuda:1}"   # for cart_iso
# -----------------------------------------------------------------------------

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

DATA_DIRS=(
  /data/wav48_silence_trimmed
  /data/fsd50k/FSD50K.dev_audio
  /data/fma_small
)
# NOTE: the directory is "preprocessed_cache" (with the "-ed"), matching
# DEFAULT_PREPROCESSED_CACHE_DIR in train_toy.py, hparam_search_driver.py and
# toy_evaluation/data.py.  A misspelled "preprocess_cache" here silently misses
# the existing manifest and triggers a full ~137k-file rebuild (~4 h).
PREPROCESS_CACHE_DIR="${PREPROCESS_CACHE_DIR:-/data/preprocessed_cache}"
CACHE_MANIFEST="$PREPROCESS_CACHE_DIR/sr16000/manifest.json"

# Common flags — IDENTICAL for both cartesian arms and identical to
# transformer_univ_full, except --universal_sampler (which defaults to
# "polar" for the reused checkpoint).  Same seed (42) so the two new models
# see the same init, data order and val split as the polar reference; the
# only thing that varies is the cone-density sampler.
COMMON=(
  --data_dir "${DATA_DIRS[@]}"
  --sample_rate 16000
  --model_type transformer
  --no_use_latent_mixup
  --use_universal_mix
  --use_soft_norm
  --use_gan
  --latent_dim 64
  --patch_size 256
  --epochs 300
  --batch_size 64
  --lr 1e-4
  --scheduler plateau
  --patience 15
  --min_delta 1e-5
  --seed 42
)
# NOTE: --use_encoder_cone is NOT passed (its default is False).  Kept OFF on
# purpose for this decoder-only ablation (see DESIGN header): encoder_cone_loss
# has its own polar sampler, independent of --universal_sampler, which would
# muddy the decoder-only signal.  --use_cycle / --use_morph_gan likewise OFF
# (defaults False).

CKPT_DIR="$REPO/toy_training/checkpoints"
mkdir -p "$CKPT_DIR"

# ---- verify the polar reference checkpoint exists ---------------------------
POLAR_REF="$CKPT_DIR/transformer_univ_full_best.pt"
if [[ ! -f "$POLAR_REF" ]]; then
  echo "ERROR: polar reference checkpoint not found: $POLAR_REF"
  echo "This script reuses transformer_univ_full_best.pt as the polar arm."
  echo "Either train it first, or point POLAR_REF at an equivalent checkpoint."
  exit 1
fi
echo "Polar arm (reused): $POLAR_REF"

# ---- verify the preprocessed cache exists (fail fast, never rebuild) --------
# train_toy.py only skips preprocessing when <cache>/sr<rate>/manifest.json is
# present; otherwise it walks the datasets and re-encodes every audio file.
# Catch a wrong --preprocess_cache_dir here rather than burning ~4 h of GPU-idle
# time per arm (and having both arms race on the same cache directory).
if [[ ! -f "$CACHE_MANIFEST" ]]; then
  echo "ERROR: preprocessed cache manifest not found: $CACHE_MANIFEST"
  echo "Refusing to start: train_toy.py would re-run full preprocessing."
  echo
  echo "Available caches under $(dirname "$PREPROCESS_CACHE_DIR"):"
  ls -d "$(dirname "$PREPROCESS_CACHE_DIR")"/*cache*/ 2>/dev/null || echo "  (none found)"
  echo
  echo "Set PREPROCESS_CACHE_DIR to the correct base directory, e.g.:"
  echo "  PREPROCESS_CACHE_DIR=/path/to/preprocessed_cache $0"
  exit 1
fi
echo "Preprocessed cache: $CACHE_MANIFEST"

PIDS=()          # PIDs of the launched arms
ARM_NAMES=()     # matching run names, for the failure report

run_arm () {
  local SAMPLER="$1"
  local DEVICE="$2"
  local NAME="ablation_sampler_${SAMPLER}"
  local LOG="$CKPT_DIR/train_${NAME}.log"
  echo "============================================================"
  echo "  Launching arm sampler=$SAMPLER on $DEVICE"
  echo "  run_name=$NAME"
  echo "  log: $LOG"
  echo "============================================================"
  # Background + nohup so it survives shell exit; the two arms run in parallel
  # on the two GPUs.  We wait on both PIDs below.
  nohup python toy_training/train_toy.py \
    --run_name "$NAME" --universal_sampler "$SAMPLER" --device "$DEVICE" \
    --preprocess_cache_dir "$PREPROCESS_CACHE_DIR" \
    "${COMMON[@]}" \
    > "$LOG" 2>&1 &
  PIDS+=("$!")
  ARM_NAMES+=("$NAME")
  echo "  PID: $!"
  echo "$NAME" >> "$CKPT_DIR/sampler_ablation_runs.txt"
}

# Launch the two cartesian arms in parallel on the two GPUs.
run_arm cart_square "$DEVICE_A"
echo
run_arm cart_iso    "$DEVICE_B"

echo
echo "Both cartesian arms launched in background. Monitor with:"
echo "  tail -f $CKPT_DIR/train_ablation_sampler_cart_square.log"
echo "  tail -f $CKPT_DIR/train_ablation_sampler_cart_iso.log"
echo
echo "The three checkpoints for the sampler ablation are:"
echo "  polar      : $POLAR_REF   (reused -- already trained)"
echo "  cart_square: $CKPT_DIR/ablation_sampler_cart_square_best.pt"
echo "  cart_iso   : $CKPT_DIR/ablation_sampler_cart_iso_best.pt"
echo
echo "Waiting for both cartesian arms to finish..."
# Wait on each PID individually and propagate failures.  A bare `wait` returns 0
# whenever the shell has no jobs left, so a crashed arm used to look like a
# successful run (observed: both arms died on a DataLoader SemLock
# PermissionError and the script still exited 0).
FAILED=()
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    FAILED+=("${ARM_NAMES[$i]}")
  fi
done

if (( ${#FAILED[@]} > 0 )); then
  echo
  echo "ERROR: ${#FAILED[@]} arm(s) failed: ${FAILED[*]}"
  echo "Check the logs, e.g.: tail -n 50 $CKPT_DIR/train_${FAILED[0]}.log"
  exit 1
fi

echo "Done. Next: score the three checkpoints on the same held-out set with"
echo "the toy_evaluation battery (mix-recon ratio, path-MSE, latent convexity)."
echo "Expected: polar ~= cart_iso  and  cart_square < polar  (docs §8.4)."
