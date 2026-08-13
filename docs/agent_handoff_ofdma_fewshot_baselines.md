# OFDMA Few-Shot Baseline Supplement Handoff

> **历史补充协议**：本文档记录的是旧官方随机划分实验，不是当前 OFDMA 主实验。
> 当前主实验使用 target-scene cold-start 协议，详见
> `docs/research/ofdma_target_scene_coldstart_protocol.md` 和
> `analysis_outputs/20260728_ofdma_patch_rank3_formal_test/README.md`。

Date: 2026-07-18

## Goal

Complete the missing baselines under the same normal-only OFDMA 1/2/4-shot
protocol used by `tools/eval_cls_ofdma_fewshot_comparison.py`.

## Fixed Protocol

- One shot is one normal scene containing all 21 SU spectrograms.
- Nested support scenes are selected from the official 8,800-scene normal
  training pool with seed 111.
- Test set is fixed: 1,000 normal scenes and 500 scenes for each of barrage,
  deceptive, pilot, sweep and random-hop interference.
- Raw `1320x70` images use the official 12-subcarrier power aggregation before
  model-specific resizing.
- Test labels are used only after score generation.
- Report image, scene-mean and scene-max AUROC, AUPRC and FPR@95TPR, including
  per-jammer, macro-jammer and overall rows.

## Methods

- Existing table: PromptAD, ViT normal memory, PatchCore-style CNN
  normal memory and the proposed dual-visual fusion.
- New runs: VAE reconstruction few-shot adaptation, Deep SVDD, PaDiM, STFPM
  and WinCLIP.
- Official PatchCore is not rerun; the existing CNN normal-memory branch is the
  agreed PatchCore-style baseline.
- The OFDMA repository VAE result (91.43 scene AUROC) uses 8,800 training scenes
  plus 200 normal calibration scenes. It is a full-normal reference, not a
  strict few-shot result, and must not be placed in the 1/2/4-shot table.

## Commands

VAE:

```bash
python tools/eval_vae_cls.py \
  --protocol ofdma --ofdma-shots 1 2 4 \
  --gpu-id 0 --epochs 100 --batch-size 128 --num-workers 4 \
  --output-root analysis_outputs/20260718_ofdma_fewshot_baselines/vae
```

Deep SVDD:

```bash
python tools/eval_deepsvdd_cls.py \
  --protocol ofdma --ofdma-shots 1 2 4 \
  --gpu-id 1 --epochs 100 --batch-size 128 --num-workers 4 \
  --output-root analysis_outputs/20260718_ofdma_fewshot_baselines/deepsvdd
```

PaDiM:

```bash
python tools/eval_padim_cls.py \
  --protocol ofdma --ofdma-shots 1 2 4 \
  --gpu-id 0 --batch-size 64 --num-workers 4 \
  --output-root analysis_outputs/20260718_ofdma_fewshot_baselines/padim
```

STFPM:

```bash
python tools/eval_stfpm_cls.py \
  --protocol ofdma --ofdma-shots 1 2 4 \
  --gpu-id 1 --epochs 50 --batch-size 64 --num-workers 4 \
  --output-root analysis_outputs/20260718_ofdma_fewshot_baselines/stfpm
```

WinCLIP:

```bash
python tools/eval_winclip_main.py \
  --protocol ofdma --ofdma-shots 1 2 4 \
  --gpu-id 0 --batch-size 32 --gallery-batch-size 16 --num-workers 4 \
  --output-root analysis_outputs/20260718_ofdma_fewshot_baselines/winclip
```

After all jobs finish:

```bash
python tools/summarize_ofdma_fewshot_baselines.py \
  --existing-results analysis_outputs/20260715_ofdma_fewshot_comparison/results.csv \
  --input vae_fewshot=analysis_outputs/20260718_ofdma_fewshot_baselines/vae \
  --input deep_svdd=analysis_outputs/20260718_ofdma_fewshot_baselines/deepsvdd \
  --input padim=analysis_outputs/20260718_ofdma_fewshot_baselines/padim \
  --input stfpm=analysis_outputs/20260718_ofdma_fewshot_baselines/stfpm \
  --input winclip=analysis_outputs/20260718_ofdma_fewshot_baselines/winclip \
  --output analysis_outputs/20260718_ofdma_fewshot_baselines/results.csv
```

## Verification

All six adapters, including the optional official PatchCore adapter, passed a
one-scene end-to-end smoke. The unified score summarizer produced 126 smoke
rows. Smoke metrics are connectivity checks only and must not be reported as
performance results.
