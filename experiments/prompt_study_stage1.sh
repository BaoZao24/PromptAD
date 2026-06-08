#!/bin/bash
# Stage 1: chirp_signal small-scale prompt study
# 7 configs on WeaponMuseum_spectrum, m10db

set -euo pipefail

DATASET="chirp_signal"
CLASS="WeaponMuseum_spectrum"
NOISE="m10db"
BASE_DIR="./result/prompt_study_stage1"

declare -A CONFIGS
CONFIGS=(
  ["01_baseline_rf"]="rf single 1"
  ["02_rf_signal_structured"]="rf_signal_structured single 1"
  ["03_grouped_mean"]="rf grouped_mean 1"
  ["04_grouped_meanmax"]="rf grouped_meanmax 1"
  ["05_grouped_softmax"]="rf grouped_softmax 1"
  ["06_n_ctx_ab_2"]="rf single 2"
  ["07_n_ctx_ab_4"]="rf single 4"
)

for name in "${!CONFIGS[@]}"; do
  read prompt_mode text_proto n_ctx_ab <<< "${CONFIGS[$name]}"
  echo "===== Running: $name | prompt=$prompt_mode | text_proto=$text_proto | n_ctx_ab=$n_ctx_ab ====="
  python train_cls.py \
    --dataset "$DATASET" \
    --class_name "$CLASS" \
    --k-shot 1 \
    --split-mode normal_75_25 \
    --noise-level "$NOISE" \
    --input-mode morph_fusion_gray_residual_a01 \
    --prompt-mode "$prompt_mode" \
    --text-prototype-mode "$text_proto" \
    --n_ctx_ab "$n_ctx_ab" \
    --cls-score-mode text_only \
    --seed 111 \
    --root-dir "${BASE_DIR}/${name}" \
    --vis False
  echo "===== Done: $name ====="
  echo ""
done

echo "All experiments completed."
