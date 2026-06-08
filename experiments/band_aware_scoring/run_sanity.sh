#!/bin/bash
# Sanity check: Band-Aware Multi-Scale Scoring
# Phase 1: Run baseline text_only trainings on 3 datasets
# Phase 2: Run band-aware scoring analysis on each checkpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

SEED=111
EPOCHS=50
SCENE="Playground_spectrum"
NOISE="m30db"
INPUT_MODE="morph_fusion_gray_residual_a01"
PROMPT_MODE="rf"
CLS_SCORE_MODE="text_only"
SPLIT_MODE="normal_75_25"
ROOT_DIR="./result"

echo "============================================"
echo "Band-Aware Scoring Sanity Check"
echo "Scene: $SCENE  |  Noise: $NOISE  |  Seed: $SEED"
echo "Input: $INPUT_MODE  |  Prompt: $PROMPT_MODE"
echo "============================================"

# Phase 1: Run baseline trainings
echo ""
echo "Phase 1: Running baseline text_only trainings..."
echo ""

# Run 3 datasets in parallel on GPUs 0, 1, 2
DATASETS=("burst_signal" "chirp_signal" "dsss_signal")
GPUS=(0 1 2)
PIDS=()

for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    GPU="${GPUS[$i]}"
    LOG="/tmp/band_baseline_${DS}_${SCENE}_${NOISE}.log"
    echo "Starting ${DS} on GPU ${GPU}..."
    MKL_THREADING_LAYER=GNU python train_cls.py \
        --dataset "${DS}" --class_name "${SCENE}" \
        --noise-level "${NOISE}" --Epoch ${EPOCHS} --gpu-id ${GPU} \
        --seed ${SEED} --k-shot 1 \
        --prompt-mode "${PROMPT_MODE}" --input-mode "${INPUT_MODE}" \
        --cls-score-mode "${CLS_SCORE_MODE}" \
        --split-mode "${SPLIT_MODE}" --normal-train-ratio 0.75 \
        --vis False --root-dir "${ROOT_DIR}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "  PID: $!  |  Log: ${LOG}"
done

echo ""
echo "Waiting for all baseline trainings to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "  PID $pid completed"
done

echo ""
echo "Baseline trainings complete!"

# Print baseline results
echo ""
echo "============================================"
echo "Baseline Results (text_only)"
echo "============================================"
for DS in "${DATASETS[@]}"; do
    CSV="${ROOT_DIR}/${DS}/${SCENE}/${NOISE}/${SPLIT_MODE}/k_1/csv/Seed_${SEED}-results.csv"
    if [ -f "$CSV" ]; then
        VAL=$(python -c "import pandas as pd; df=pd.read_csv('${CSV}', index_col=0); print(round(float(df.loc['${DS}-${SCENE}', 'i_roc']), 4))")
        echo "  ${DS}: Image-AUROC = ${VAL}"
    else
        echo "  ${DS}: No CSV found at ${CSV}"
    fi
done

# Phase 2: Run band-aware scoring analysis
echo ""
echo "============================================"
echo "Phase 2: Band-Aware Scoring Analysis"
echo "============================================"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "--- ${DS} ---"
    python tools/evaluate_band_aware_scoring.py \
        --dataset "${DS}" --class_name "${SCENE}" \
        --noise-level "${NOISE}" --seed ${SEED} \
        --prompt-mode "${PROMPT_MODE}" --input-mode "${INPUT_MODE}" \
        --root-dir "${ROOT_DIR}" \
        --versions A B C \
        --betas 0.02 0.05 0.1 \
        --topk-ratios 0.05 0.1 0.2 \
        --save-npz "${SCRIPT_DIR}/visual_maps_${DS}_${SCENE}_${NOISE}.npz"
done

echo ""
echo "============================================"
echo "Sanity check complete!"
echo "============================================"
