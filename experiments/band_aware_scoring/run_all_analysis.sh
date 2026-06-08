#!/bin/bash
# Run band-aware scoring analysis on all 3 datasets
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

SCENE="Playground_spectrum"
NOISE="m30db"
ROOT="./result"
SPLIT="normal_75_25"

SUMMARY_FILE="${SCRIPT_DIR}/sanity_results.csv"
echo "dataset,version,variant,bands,topk,agg,calib,beta,auroc,delta" > "$SUMMARY_FILE"

for DS in burst_signal chirp_signal dsss_signal; do
    SCORES_NPZ="${ROOT}/${DS}/${SCENE}/${NOISE}/${SPLIT}/k_1/scores/Seed_111-image_scores.npz"
    NORMAL_MAPS="${SCRIPT_DIR}/normal_maps_${DS}_${SCENE}_${NOISE}.npy"

    if [ ! -f "$SCORES_NPZ" ]; then
        echo "SKIP ${DS}: no scores npz at ${SCORES_NPZ}"
        continue
    fi

    NORMAL_FLAG=""
    if [ -f "$NORMAL_MAPS" ]; then
        NORMAL_FLAG="--load-normal-maps $NORMAL_MAPS"
    fi

    echo ""
    echo "============================================"
    echo "Analyzing ${DS}"
    echo "============================================"

    # Run analysis and capture results
    python tools/evaluate_band_aware_scoring.py \
        --dataset "${DS}" --class_name "${SCENE}" \
        --noise-level "${NOISE}" --seed 111 \
        --prompt-mode rf --input-mode morph_fusion_gray_residual_a01 \
        --root-dir "${ROOT}" --gpu-id 0 \
        --versions A B C \
        --betas 0.02 0.05 0.1 \
        --topk-ratios 0.05 0.1 0.2 \
        --load-training-scores "${SCORES_NPZ}" \
        ${NORMAL_FLAG} \
        2>&1 | tee "${SCRIPT_DIR}/analysis_${DS}.log"

    # Extract best results to summary
    echo "  Done."
done

echo ""
echo "============================================"
echo "All analyses complete!"
echo "============================================"

# Print summary
for DS in burst_signal chirp_signal dsss_signal; do
    LOG="${SCRIPT_DIR}/analysis_${DS}.log"
    if [ -f "$LOG" ]; then
        echo ""
        echo "=== ${DS} ==="
        grep "BASELINE" "$LOG" | head -1
        grep "TOP RESULTS" -A 5 "$LOG" | tail -5
    fi
done
