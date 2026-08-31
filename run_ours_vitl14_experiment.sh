#!/usr/bin/env bash
# Ours: replace the frozen ViT-B-16-plus-240 branch with ViT-L-14.
# The CNN branch, normal memories, gates, manifests, and random seed follow
# the current protocol in docs/paper/现有方案介绍.md.
set -euo pipefail

source activate prompt_ad 2>/dev/null || conda activate prompt_ad
cd /mnt/data/wangbei/SpectraMemAD

ROOT=analysis_outputs/exploratory/20260824_ours_vitl14_replace
GPU=2
CKPT=analysis_outputs/02_current_baselines/promptad_formal_baseline/pooled_rf_rgb_cls/checkpoint/overall-best.pt
IH_MAN=analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json
CNN_ROOT=analysis_outputs/exploratory/20260819_ours_wrn50_replace

run_stage() {
  local name="$1"
  shift
  if [[ -f "$ROOT/stage_${name}.done" ]]; then
    echo "[skip] ${name} already done"
    return 0
  fi
  echo "===== [$(date '+%F %T')] stage ${name} ====="
  "$@" >"$ROOT/logs/${name}.log" 2>&1
  touch "$ROOT/stage_${name}.done"
  tail -5 "$ROOT/logs/${name}.log"
}

mkdir -p "$ROOT/logs"

# ---------- In-house RF ----------
run_stage ih_vit python tools/eval_cls_vit_patchcore_gallery.py \
  --output-root "$ROOT/inhouse_vit" \
  --support-manifest "$IH_MAN" --normal-sampling per_frequency \
  --checkpoint "$CKPT" \
  --backbone ViT-L-14 --pretrained_dataset laion400m_e32 \
  --img-resize 224 --img-cropsize 224 \
  --coreset-method farthest --coreset-ratio 0.5 \
  --nn-topk 5 --nn-agg mean \
  --paired-tta rf_spectral_response_v1 --paired-tta-fusion max \
  --paired-tta-memory-layout merged \
  --batch-size 64 --num-workers 4 --gpu-id "$GPU" --seed 111

run_stage ih_vit_ref python tools/eval_cls_vit_patchcore_gallery.py \
  --output-root "$ROOT/inhouse_vit_reference" \
  --support-manifest "$IH_MAN" --normal-sampling per_frequency \
  --checkpoint "$CKPT" \
  --backbone ViT-L-14 --pretrained_dataset laion400m_e32 \
  --img-resize 224 --img-cropsize 224 \
  --coreset-method farthest --coreset-ratio 0.5 \
  --nn-topk 5 --nn-agg mean \
  --paired-tta rf_spectral_response_v1 --paired-tta-fusion max \
  --paired-tta-memory-layout merged \
  --batch-size 8 --num-workers 0 --support-reference-only \
  --gpu-id "$GPU" --seed 111

run_stage ih_fusion_r18 python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol rf_target --gate-protocol support_only \
  --vit-score-dir "$ROOT/inhouse_vit/scores" \
  --cnn-score-dir "$CNN_ROOT/inhouse_cnn_r18/scores" \
  --vit-reference-dir "$ROOT/inhouse_vit_reference" \
  --cnn-reference-dir "$CNN_ROOT/inhouse_cnn_r18_reference" \
  --output-root "$ROOT/inhouse_fusion_r18"

# ---------- Public RF k=1/2/4 ----------
for K in 1 2 4; do
  MAN=analysis_outputs/20260810_public_rf_k_per_frequency/k${K}/support_manifest.json

  run_stage "pub_vit_k${K}" python tools/eval_cls_public_rf_vit_patchcore_gallery.py \
    --output-root "$ROOT/public_vit_k${K}" \
    --normal-sampling per_frequency \
    --support-manifest "$MAN" --support-seed 111 \
    --checkpoint "$CKPT" \
    --backbone ViT-L-14 --pretrained_dataset laion400m_e32 \
    --img-resize 224 --img-cropsize 224 \
    --batch-size 96 --num-workers 4 --gpu-id "$GPU" \
    --paired-tta rf_spectral_response_v1 --paired-tta-fusion max \
    --paired-tta-memory-layout merged \
    --nn-topk 5 --nn-agg mean --coreset-ratio 0.5 --coreset-method farthest \
    --seed 111

  run_stage "pub_vit_ref_k${K}" python tools/eval_cls_public_rf_vit_patchcore_gallery.py \
    --output-root "$ROOT/public_vit_reference_k${K}" \
    --normal-sampling per_frequency \
    --support-manifest "$MAN" --support-seed 111 \
    --checkpoint "$CKPT" \
    --backbone ViT-L-14 --pretrained_dataset laion400m_e32 \
    --img-resize 224 --img-cropsize 224 \
    --batch-size 8 --num-workers 0 --gpu-id "$GPU" \
    --paired-tta rf_spectral_response_v1 --paired-tta-fusion max \
    --paired-tta-memory-layout merged \
    --nn-topk 5 --nn-agg mean --coreset-ratio 0.5 --coreset-method farthest \
    --support-reference-only --seed 111

  run_stage "pub_fusion_r18_k${K}" python tools/eval_cls_dual_visual_evidence_fusion.py \
    --protocol public_rf --gate-protocol support_only \
    --vit-score-dir "$ROOT/public_vit_k${K}/scores" \
    --cnn-score-dir "$CNN_ROOT/public_cnn_resnet18_k${K}/scores" \
    --vit-reference-dir "$ROOT/public_vit_reference_k${K}" \
    --cnn-reference-dir "$CNN_ROOT/public_cnn_resnet18_reference_k${K}" \
    --support-manifest "$MAN" \
    --output-root "$ROOT/public_fusion_r18_k${K}"
done

# ---------- OFDMA ----------
run_stage ofdma_l14 python tools/eval_cls_ofdma_target_scene_ours.py \
  --dataset-root /mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic \
  --output-root "$ROOT/ofdma_l14" \
  --split test --shots 1 2 4 \
  --backbone ViT-L-14 --img-resize 224 --img-cropsize 224 \
  --vit-tta-ablation spectral_response \
  --checkpoint "$CKPT" --batch-size 32 --num-workers 4 --gpu-id "$GPU"

# ---------- FedJam ----------
run_stage fedjam_l14 env CUDA_VISIBLE_DEVICES="$GPU" python tools/eval_fedjam_fewshot_dual.py \
  --data-root /mnt/data/wangbei/data/FedJam \
  --output-root "$ROOT/fedjam_l14" \
  --checkpoint "$CKPT" --backbone ViT-L-14 \
  --img-resize 224 --img-cropsize 224 \
  --batch-size 8 --seed 111 --gpu-id "$GPU" \
  --vit-tta rf_spectral_response_v1 --vit-tta-memory-layout merged

echo "===== [$(date '+%F %T')] ALL DONE ====="
