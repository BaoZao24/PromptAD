#!/bin/bash
# Phase 2: Small-scale sanity experiment
# Scene: Playground_spectrum | Noise: m30db | Seed: 111 | Epochs: 50
# Compares: baseline (gray_residual_a01) vs 方案A vs 方案B

set -e

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate prompt_ad

SCENE="Playground_spectrum"
NOISE="m30db"
SEED=111
EPOCHS=50
ROOT_DIR="./result"

DATASETS=("burst_signal" "chirp_signal" "dsss_signal")
METHODS=("morph_fusion_gray_residual_a01" "morph_fusion_normal_bg_residual" "morph_fusion_contrast_residual_normal_bg")
METHOD_LABELS=("baseline" "planA_normal_bg" "planB_contrast_normal_bg")

RESULT_FILE="experiments/normal_bg_deviation/sanity_results.csv"
echo "dataset,method,image_auroc" > "$RESULT_FILE"

for ds_idx in "${!DATASETS[@]}"; do
    ds="${DATASETS[$ds_idx]}"
    for m_idx in "${!METHODS[@]}"; do
        method="${METHODS[$m_idx]}"
        label="${METHOD_LABELS[$m_idx]}"

        echo ""
        echo "============================================================"
        echo "Running: $ds | $label ($method)"
        echo "============================================================"

        python train_cls.py \
            --dataset "$ds" \
            --class_name "$SCENE" \
            --noise-level "$NOISE" \
            --split-mode normal_75_25 \
            --seed "$SEED" \
            --Epoch "$EPOCHS" \
            --prompt-mode rf \
            --cls-score-mode text_only \
            --input-mode "$method" \
            --root-dir "$ROOT_DIR" \
            2>&1 | tee "experiments/normal_bg_deviation/${ds}_${label}.log"

        # Extract best Image-AUROC from log
        best_auroc=$(grep "Image-AUROC:" "experiments/normal_bg_deviation/${ds}_${label}.log" | tail -1 | grep -oP 'Image-AUROC:\K[0-9.]+')
        echo "$ds,$label,$best_auroc" >> "$RESULT_FILE"
        echo "Best Image-AUROC for $ds/$label: $best_auroc"
    done
done

echo ""
echo "============================================================"
echo "Phase 2 experiments complete. Results:"
echo "============================================================"
cat "$RESULT_FILE"
