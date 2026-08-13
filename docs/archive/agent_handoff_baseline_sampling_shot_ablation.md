# Agent Handoff: Generic Baseline Sampling Shot Ablation

Date: 2026-07-06

## Objective

补齐 **PromptAD generic baseline** 在不同 normal support 采样下的历史结果，覆盖当时的三个数据集
（不含后来加入的 FedJam；本 handoff 不代表当前论文主表）：

```text
self RF
public RF
Spectrum
```

采样设置：

```text
self RF / public RF: per_frequency, 1shot, 2shot, 4shot
Spectrum: 1shot, 2shot, 4shot
```

注意：当前方案的 `1shot/2shot/4shot/per_frequency` 已经完成，结果在：

```text
analysis_outputs/20260706_sampling_shot_ablation/
```

本 handoff 只跑 baseline。

## Baseline Definition

严格 baseline 使用通用 PromptAD prompt，不使用 RF 优化提示词：

```text
PromptAD generic baseline
= generic text_score + ViT patch anomaly map
= harmonic(generic text_score, max(ViT patch anomaly map))
```

所有命令必须显式使用：

```bash
--prompt-mode generic
--input-mode rgb
--text-prototype-mode single
--cls-score-mode text_only
```

不要把当前方案里的 `ViT Normal Gallery` 单分支当成 baseline。

## Output Root

```text
analysis_outputs/20260706_baseline_sampling_shot_ablation/
```

目录结构：

```text
analysis_outputs/20260706_baseline_sampling_shot_ablation/{sampling}/
  self_baseline/
  public_baseline/
  spectrum_baseline/   # only for 1shot/2shot/4shot
  self_baseline.log
  public_baseline.log
  spectrum_baseline.log
```

汇总文件：

```text
analysis_outputs/20260706_baseline_sampling_shot_ablation/
  baseline_sampling_shot_ablation_summary.csv
  README.md
  scheduler.log
```

## Recommended Command

在 tmux 里从仓库根目录运行：

```bash
python run_baseline_sampling_ablation_scheduler.py
```

该脚本会跑 11 个 job：

```text
self RF: 4 jobs
public RF: 4 jobs
Spectrum: 3 jobs
```

脚本是幂等的：如果某个 `{sampling}/{role}/summary.json` 已存在，会跳过。

默认使用 GPU：

```text
0, 2, 3
```

如需改 GPU，编辑 `run_baseline_sampling_ablation_scheduler.py` 顶部的 `GPUS`。

## Manual Commands

如果不使用调度器，手动跑下面两条。替换：

```bash
SAMPLING=per_frequency   # or 1shot / 2shot / 4shot
GPU=0
ROOT=analysis_outputs/20260706_baseline_sampling_shot_ablation/${SAMPLING}
PUB_CKPT=analysis_outputs/02_current_baselines/promptad_formal_baseline/pooled_rf_rgb_cls/checkpoint/overall-best.pt
```

### Self RF baseline

```bash
python tools/eval_cls_vit_patch_gallery.py \
  --output-root ${ROOT}/self_baseline \
  --normal-sampling ${SAMPLING} \
  --prompt-mode generic \
  --input-mode rgb \
  --text-prototype-mode single \
  --cls-score-mode text_only \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/self_baseline.log
```

主指标：

```text
harmonic_text_vit_max_auc_macro
```

### Public RF baseline

```bash
python tools/eval_cls_public_rf_vit_patchcore_gallery.py \
  --output-root ${ROOT}/public_baseline \
  --normal-sampling ${SAMPLING} \
  --checkpoint ${PUB_CKPT} \
  --prompt-mode generic \
  --input-mode rgb \
  --text-prototype-mode single \
  --cls-score-mode text_only \
  --coreset-method random \
  --coreset-ratio 1.0 \
  --nn-topk 1 \
  --nn-agg mean \
  --paired-tta none \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/public_baseline.log
```

主指标：

```text
text_vit_max_auc_macro
```

### Spectrum baseline

```bash
SAMPLING=1shot   # or 2shot / 4shot; no per_frequency for Spectrum

python tools/eval_cls_spectrum_vit_nn_gallery.py \
  --output-root ${ROOT}/spectrum_baseline \
  --normal-sampling ${SAMPLING} \
  --checkpoint ${PUB_CKPT} \
  --prompt-mode generic \
  --input-mode rgb \
  --text-prototype-mode single \
  --cls-score-mode text_only \
  --coreset-method random \
  --coreset-ratio 1.0 \
  --nn-topk 1 \
  --nn-agg mean \
  --paired-tta none \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/spectrum_baseline.log
```

主指标：

```text
text_vit_max_auc_macro
```

注意：Spectrum 文件名没有 RF 频段标注，所以不跑 `per_frequency`。Spectrum 只跑 `1shot/2shot/4shot`。

## Why Rerun per_frequency

已有目录：

```text
analysis_outputs/20260706_public_text_vit_baseline_generic_frequency_one_per_band/
analysis_outputs/20260706_self_text_vit_baseline_generic_frequency_one_per_band/
```

但旧 summary 没记录 `prompt_mode`，而相关脚本默认值是 `rf`。为了保证 baseline 口径干净，本次 baseline ablation 要把 `per_frequency` 也一起重跑，并显式传 `--prompt-mode generic`。

## Expected Comparison

完成后对比两张表：

```text
analysis_outputs/20260706_baseline_sampling_shot_ablation/baseline_sampling_shot_ablation_summary.csv
analysis_outputs/20260706_sampling_shot_ablation/sampling_shot_ablation_summary.csv
```

重点看：

- baseline 在 `1shot/2shot/4shot/per_frequency` 下怎么变；Spectrum 只看 `1shot/2shot/4shot`。
- 当前方案是否在同一 sampling 下超过 baseline。
- `per_frequency` 的优势是否同时存在于 baseline 和当前方案。
- Spectrum 没有频段标注，不做 `per_frequency`。
