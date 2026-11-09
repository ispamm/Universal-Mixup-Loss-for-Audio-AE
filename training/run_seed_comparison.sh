#!/usr/bin/env bash
# =============================================================================
# Seed-to-seed comparison — universal_mix recipe at the canonical 16 kHz geometry
# (latent_dim=64, patch_size=256, sr=16000 → 4:1 per-scalar compression,
#  62.5 latents/s).
#
# Purpose: quantify seed variance of the Transformer autoencoder trained
# with the "universal cone" additivity loss, so headline metrics (recon, LIMS
# decoder additivity, encoder homomorphism, latent convexity) can be reported
# as mean ± std over 3 seeds instead of a single point estimate.
#
# Seeds: 42 (already trained — `transformer_univ_full`, ckpt
#        transformer_univ_full_best.pt), 43, 44 (this script).
# All three use the SAME val split (deterministic 90/10 via seed 42 in
# train.py's get_train_val_track_split) and the SAME eval pair sets
# (every evaluation script fixes its own pair-generation seed), so the
# three checkpoints are scored on identical held-out data — the only thing
# that varies is the training RNG.
#
# Recipe (matches transformer_univ_full exactly, from its wandb config):
#   universal_mix (cone) loss ON, latent_mixup OFF, soft-norm KL ON,
#   GAN ON (λ_adv 0.05), LeakyReLU + 2 final Snake decoder layers,
#   scale-sensitive bottleneck (final LayerNorm OFF, auto via universal_mix),
#   AdamW lr=1e-4 wd=1e-4, plateau scheduler, patience 15, min_delta 1e-5,
#   batch 64, 2 s chunks, 50.6 M params.
#
# Reference seed-42 result (transformer_univ_full, early-stopped ep ~218):
#   val_total ≈ 1.099, val_mag ≈ 0.536, val_univ ≈ 0.563, loss_D ≈ 0.108
# =============================================================================
set -euo pipefail

# ---- EDIT THIS to your free GPU (recent runs used cuda:3) -------------------
DEVICE="${DEVICE:-cuda:0}"
# -----------------------------------------------------------------------------

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

DATA_DIRS=(
  /data/wav48_silence_trimmed
  /data/fsd50k/FSD50K.dev_audio
  /data/fma_small
)

# Common flags — identical for every seed (only --seed and --run_name change).
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
  --device "$DEVICE"
)

CKPT_DIR="$REPO/training/checkpoints"
mkdir -p "$CKPT_DIR"

run_seed () {
  local SEED="$1"
  local NAME="univ_code64_s${SEED}"
  local LOG="$CKPT_DIR/train_${NAME}.log"
  echo "============================================================"
  echo "  Launching seed $SEED  →  run_name=$NAME"
  echo "  log: $LOG"
  echo "  device: $DEVICE"
  echo "============================================================"
  # nohup so it survives shell exit; logs go to the checkpoints dir, matching
  # the naming convention of every other run in this repo.
  nohup python training/train.py \
    --run_name "$NAME" --seed "$SEED" \
    "${COMMON[@]}" \
    > "$LOG" 2>&1 &
  echo "  PID: $!"
  echo "$NAME" >> "$CKPT_DIR/seed_comparison_runs.txt"
}

# Run the two missing seeds. Seed 42 is already trained (transformer_univ_full).
run_seed 43
echo
run_seed 44

echo
echo "Both seeds launched. Monitor with:"
echo "  tail -f $CKPT_DIR/train_univ_code64_s43.log"
echo "  tail -f $CKPT_DIR/train_univ_code64_s44.log"
echo
echo "When both finish, the three checkpoints for the seed comparison are:"
echo "  seed 42: $CKPT_DIR/transformer_univ_full_best.pt   (already exists)"
echo "  seed 43: $CKPT_DIR/univ_code64_s43_best.pt"
echo "  seed 44: $CKPT_DIR/univ_code64_s44_best.pt"
echo
echo "Then evaluate all three on the same held-out set with the evaluation"
echo "battery."
