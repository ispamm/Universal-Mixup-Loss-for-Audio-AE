#!/usr/bin/env bash
# =============================================================================
# Encoder-Decoder cone alignment WITHOUT latent space collapse — 2-GPU plan
#
# Two approaches, one GPU each, both at the canonical 16 kHz geometry
# (latent_dim=64, patch_size=256, sr=16000 → 4:1 per-scalar compression).
#
# Baseline reference (already trained, do NOT re-run):
#   transformer_univ_full_best    dec cone only      enc_mse@0.5=0.167  eff-rank=41.4
#   transformer_mixup_enc_best    dec+enc cone       enc_mse@0.5=0.020  eff-rank=39.4 (collapsed!)
#   transformer_softMax_full_best no cone            enc_mse@0.5=0.893  eff-rank=106.6
#
# SUCCESS = eff-rank ≥ 80 AND enc_mse@0.5 ≤ 0.20
# =============================================================================
set -euo pipefail

# ---- EDIT THESE to your free GPUs (recent runs used cuda:1 and cuda:3) ----
GPU_A="${GPU_A:-cuda:1}"
GPU_B="${GPU_B:-cuda:3}"
# ---------------------------------------------------------------------------

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

DATA_DIRS=(
  /data/wav48_silence_trimmed
  /data/fsd50k/FSD50K.dev_audio
  /data/fma_small
)

# Common flags — identical base config for both approaches (only the
# alignment mechanism differs).
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
  --adv_loss_weight 0.05
  --universal_mix_weight 1.0
)

CKPT_DIR="$REPO/toy_training/checkpoints"
mkdir -p "$CKPT_DIR"

# =============================================================================
#  Approach A — Annealed encoder cone + boosted KL anti-collapse  (GPU A)
#
#  The encoder_cone_loss is an MSE on the raw code geometry that can collapse
#  the latent to a low-rank subspace.  Two mitigations:
#    1. --encoder_cone_anneal_fraction 0.25: cosine-anneal λ_enc from 0.5 to 0
#       over the last 25% of training (the never-tested flag).  The encoder
#       learns the cone geometry early, then the KL re-spreads the codes.
#    2. --soft_norm_kl_weight 2.0 (up from 1.0): stronger isotropy pressure
#       so the KL can compete with the enc-cone collapse force.
#    3. --encoder_cone_weight 0.5 (down from 1.0): gentler collapse pressure.
# =============================================================================
NAME_A="univ_enc_anneal_kl2"
LOG_A="$CKPT_DIR/train_${NAME_A}.log"
echo "============================================================"
echo "  Approach A: Annealed enc-cone + boosted KL  →  $NAME_A"
echo "  GPU: $GPU_A"
echo "  log: $LOG_A"
echo "============================================================"
nohup python toy_training/train_toy.py \
  --run_name "$NAME_A" --seed 42 \
  "${COMMON[@]}" \
  --use_encoder_cone --encoder_cone_mode auto \
  --encoder_cone_weight 0.5 --encoder_cone_anneal_fraction 0.25 \
  --soft_norm_kl_weight 2.0 \
  --device "$GPU_A" \
  > "$LOG_A" 2>&1 &
echo "  PID: $!"

# =============================================================================
#  Approach B — Decoder cone + cycle-consistency encoder alignment  (GPU B)
#
#  Instead of directly penalizing encoder linearity (encoder_cone_loss, which
#  collapses the latent), enforce encoder-decoder alignment INDIRECTLY through
#  the round-trip: E(D(z_mix)) ≈ z_mix.  The encoder is pushed toward cone
#  geometry because it must produce codes whose cone-mixtures round-trip through
#  the (cone-trained) decoder — but there is no MSE on the raw code geometry,
#  so there is no incentive to reduce rank.
#
#  Uses the new --use_cycle flag (decoupled from --use_morph_gan):
#    - No morph-gan adversarial negatives (only standard spectral GAN).
#    - Cycle loss uses the universal cone mixture (A·cosθ·Z_A + A·sinθ·Z_B).
#    - Decoder is frozen during the cycle pass; gradients train the encoder.
#    - Standard soft_norm_kl_weight 1.0 (the cycle doesn't collapse, so no
#      need to boost the KL).
# =============================================================================
NAME_B="univ_cycle_enc"
LOG_B="$CKPT_DIR/train_${NAME_B}.log"
echo
echo "============================================================"
echo "  Approach B: Decoder cone + cycle-consistency  →  $NAME_B"
echo "  GPU: $GPU_B"
echo "  log: $LOG_B"
echo "============================================================"
nohup python toy_training/train_toy.py \
  --run_name "$NAME_B" --seed 42 \
  "${COMMON[@]}" \
  --use_cycle --cycle_weight 1.0 \
  --device "$GPU_B" \
  > "$LOG_B" 2>&1 &
echo "  PID: $!"

echo
echo "Both approaches launched. Monitor with:"
echo "  tail -f $LOG_A"
echo "  tail -f $LOG_B"
echo
echo "When both finish, the checkpoints are:"
echo "  A: $CKPT_DIR/${NAME_A}_best.pt"
echo "  B: $CKPT_DIR/${NAME_B}_best.pt"
echo
echo "Then evaluate with the standard battery + effective rank:"
echo "  python toy_evaluation/evaluate_all.py \\"
echo "    --checkpoints \\"
echo "      $CKPT_DIR/transformer_univ_full_best.pt \\"
echo "      $CKPT_DIR/transformer_mixup_enc_best.pt \\"
echo "      $CKPT_DIR/transformer_softMax_full_best.pt \\"
echo "      $CKPT_DIR/${NAME_A}_best.pt \\"
echo "      $CKPT_DIR/${NAME_B}_best.pt \\"
echo "    --data_dir ${DATA_DIRS[*]} \\"
echo "    --num_pairs 1000 --num_clips 256 --num_mixups 512 --num_recon_clips 200 \\"
echo "    --device cuda:0 \\"
echo "    --output_dir toy_evaluation/eval_align_nocollapse"
