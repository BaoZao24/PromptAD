# Agent Handoff: Normal Support Sampling Ablation

Date: 2026-07-06

## Objective

补充 normal support 采样方式对当前主方案的影响：

```text
per_frequency vs 1shot vs 2shot vs 4shot
```

当前主方案：

```text
ViT Normal Gallery
+ CNN Local Gallery
+ Normal-calibrated confidence fusion
```

本实验只重新生成分数和融合结果，不重新训练模型。

## Code Status

已完成代码准备：

- `utils/rf_frequency_sampling.py`
  - 新增正式采样名 `per_frequency`
  - 保留旧别名 `frequency_one_per_band`
  - 新增 `1shot / 2shot / 4shot`
- 已接入主要入口：
  - `tools/eval_cls_vit_patch_gallery.py`
  - `tools/eval_cls_vit_patchcore_gallery.py`
  - `tools/eval_cls_public_rf_vit_patchcore_gallery.py`
  - `tools/eval_patchcore_cls.py`
  - `tools/eval_cls_public_rf_dual_gallery.py`

编译检查已通过：

```bash
python -m py_compile \
  utils/rf_frequency_sampling.py \
  tools/eval_cls_vit_patch_gallery.py \
  tools/eval_cls_vit_patchcore_gallery.py \
  tools/eval_cls_public_rf_dual_gallery.py \
  tools/eval_cls_public_rf_vit_patchcore_gallery.py \
  tools/eval_patchcore_cls.py
```

## Sampling Definitions

| sampling | meaning |
|---|---|
| `per_frequency` | 当前方案名；按频段去重选择 normal support，旧名 `frequency_one_per_band` 等价 |
| `1shot` | 从候选 normal support 中确定性选择 1 张 |
| `2shot` | 从候选 normal support 中确定性选择 2 张 |
| `4shot` | 从候选 normal support 中确定性选择 4 张 |

注意：`per_frequency` 是实验采样设置，不是方法架构模块。

## Output Root

统一输出到：

```text
analysis_outputs/20260706_sampling_shot_ablation/
```

每个采样模式单独建目录：

```text
analysis_outputs/20260706_sampling_shot_ablation/{sampling}/
  self_vit/
  self_cnn/
  self_fusion/
  public_vit/
  public_cnn/
  public_fusion/
```

## Run Plan

建议在 tmux 中运行。可以按 sampling 拆给不同 agent：

- Agent A: `per_frequency`
- Agent B: `1shot`
- Agent C: `2shot`
- Agent D: `4shot`

如果 GPU 紧张，优先跑 `1shot/2shot/4shot`，`per_frequency` 可用已有结果做参考，但最好重跑保证同代码口径。

## Commands

下面命令中替换：

```bash
SAMPLING=per_frequency   # or 1shot / 2shot / 4shot
GPU=0
ROOT=analysis_outputs/20260706_sampling_shot_ablation/${SAMPLING}
```

### 1. Self RF: ViT Normal Gallery

```bash
python tools/eval_cls_vit_patchcore_gallery.py \
  --output-root ${ROOT}/self_vit \
  --normal-sampling ${SAMPLING} \
  --patch-layer concat \
  --coreset-method farthest \
  --coreset-ratio 0.5 \
  --nn-topk 5 \
  --nn-agg mean \
  --paired-tta stft_shift_blur \
  --paired-tta-fusion max \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/self_vit.log
```

### 2. Self RF: CNN Local Gallery

```bash
python tools/eval_patchcore_cls.py \
  --protocol rf_target \
  --rf-train-mode pooled \
  --output-root ${ROOT}/self_cnn \
  --normal-sampling ${SAMPLING} \
  --batch-size 32 \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/self_cnn.log
```

### 3. Self RF: Calibrated Fusion

```bash
python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol rf_target \
  --vit-score-dir ${ROOT}/self_vit/scores \
  --cnn-score-dir ${ROOT}/self_cnn/scores \
  --output-root ${ROOT}/self_fusion \
  2>&1 | tee ${ROOT}/self_fusion.log
```

### 4. Public RF: ViT Normal Gallery

```bash
python tools/eval_cls_public_rf_vit_patchcore_gallery.py \
  --output-root ${ROOT}/public_vit \
  --normal-sampling ${SAMPLING} \
  --patch-layer concat \
  --coreset-method farthest \
  --coreset-ratio 0.5 \
  --nn-topk 5 \
  --nn-agg mean \
  --paired-tta stft_shift_blur \
  --paired-tta-fusion max \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/public_vit.log
```

### 5. Public RF: CNN Local Gallery

```bash
python tools/eval_patchcore_cls.py \
  --protocol public_rf \
  --output-root ${ROOT}/public_cnn \
  --normal-sampling ${SAMPLING} \
  --batch-size 32 \
  --gpu-id ${GPU} \
  2>&1 | tee ${ROOT}/public_cnn.log
```

### 6. Public RF: Calibrated Fusion

```bash
python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol public_rf \
  --vit-score-dir ${ROOT}/public_vit/scores \
  --cnn-score-dir ${ROOT}/public_cnn/scores \
  --output-root ${ROOT}/public_fusion \
  2>&1 | tee ${ROOT}/public_fusion.log
```

## Expected Metrics

每个 fusion 目录看：

```text
summary.json
macro.normal_calibrated_confidence_or_auc
macro.vit_auc
macro.cnn_local_auc
macro.or_evidence_auc
```

最终主比较表：

| sampling | self RF ViT | self RF CNN | self RF calibrated | public RF ViT | public RF CNN | public RF calibrated |
|---|---:|---:|---:|---:|---:|---:|
| per_frequency | | | | | | |
| 1shot | | | | | | |
| 2shot | | | | | | |
| 4shot | | | | | | |

## Success Criteria

优先看：

1. `normal_calibrated_confidence_or_auc` 是否随 shot 数稳定提升。
2. `1shot/2shot/4shot` 是否能接近或超过 `per_frequency`。
3. self RF 和 public RF 是否趋势一致。
4. CNN 分支是否只在某些采样下有明显补充作用。

## Reporting Requirements

完成后请写一个汇总文件：

```text
analysis_outputs/20260706_sampling_shot_ablation/README.md
```

可以直接运行汇总脚本：

```bash
python tools/summarize_sampling_shot_ablation.py \
  --root analysis_outputs/20260706_sampling_shot_ablation
```

需要包含：

- 每个 sampling 的 self/public macro AUROC 表。
- 每个 sampling 的 selected normal 数量。
- 与当前 `per_frequency` 的差值。
- 哪个 sampling 推荐作为正式实验设置。

## Notes

- `per_frequency` 替代旧文档中的 `frequency_one_per_band` 作为正式名字。
- 旧参数 `frequency_one_per_band` 仍然可用，只为兼容历史命令。
- 不要把 `per_frequency` 画进架构图；它只是采样设置。
- 本实验不涉及 PromptAD text/prompt 分支。
