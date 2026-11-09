#!/usr/bin/env bash
# Exp 4 seed replication at 2 s chunks.
#
# 3 DiT seeds x 4 configurations (univ/softnorm x rf/regressor) = 12 runs.
# Each GPU takes one latent space and all 6 of its runs, so a partially
# completed queue is still scientifically usable (3 seeds of one latent space).
#
# At (64,125) a 50-epoch run over 50k mixtures costs ~0.4 h of pure training
# (measured 18.0 ms/step fwd+bwd vs 115.5 ms at (64,625)), plus eval overhead.
#
# Usage:  bash training/run_exp4_seeds_2s.sh <gpu> <univ|softnorm>
set -u
cd "$(dirname "$0")/.."

GPU="$1"; LATENT="$2"          # e.g. 0 univ
SEEDS="42 43 44"
MIX="evaluation/exp4_mixtures_2s/${LATENT}"
OUT="training/checkpoints/fm_cond_2s"

if [ "$LATENT" = "univ" ]; then
  AE="training/checkpoints/transformer_univ_warm_v2_best.pt"
else
  AE="training/checkpoints/transformer_softMax_full_best.pt"
fi
mkdir -p "$OUT"

for OBJ in rf mse; do
  for S in $SEEDS; do
    NAME="fm2s_${OBJ}_${LATENT}_s${S}"
    echo "=== $(date +%H:%M:%S) starting $NAME on cuda:$GPU"
    .venv/bin/python training/train_fm_cond.py \
      --mixture_dir "$MIX" --ae_checkpoint "$AE" \
      --run_name "$NAME" --device "cuda:$GPU" \
      --seed "$S" --objective "$OBJ" --cond mix \
      --epochs 50 --batch_size 32 \
      --eval_every 2 --eval_clips 32 --eval_seeds 4 \
      --sample_steps 32 --step_sweep 1,2,4,8,16,32 --val_batches 20 \
      --output_dir "$OUT" || echo "!!! $NAME FAILED (exit $?)"
  done
done
echo "=== $(date +%H:%M:%S) queue for $LATENT finished"
