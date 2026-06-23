# PromptAD RF 跨库 Dense Mask Supervision 实验计划

本文档替代旧的 A-F score-fusion / VCPA adapter 消融计划。当前主线是在 PromptAD 上新增
**dense mask supervision** 分支，并按“public source 监督训练 + target 正常样本适配 + target test
评估”的协议做跨库实验。

## 1. 这次跨库到底怎么跨

正式协议固定为：

```text
source 训练库:
  /mnt/data/wangbei/data/RF_SPE_PNG
  public RF source normal + abnormal + mask
  用途: 训练 dense mask 分支，让模型学异常区域长什么样。

target 适配库:
  /mnt/data/wangbei/data/datasets/{burst, chirp, dsss}
  project RF target train normal only
  用途: 建 target-domain 正常特征库，也就是 PromptAD 原本需要的正常 gallery。

target 测试库:
  /mnt/data/wangbei/data/datasets/{burst, chirp, dsss}
  project RF target test normal + abnormal + mask
  用途: 最终评估 image AUROC / pixel AUROC / pixel AP。
```

这不是单库作弊，因为：

```text
不会使用 target train abnormal
不会使用 target train mask
不会使用 target test 参与训练或建库
target train normal 只用于正常库/normal adaptation
```

也就是说，方法不是 `public source -> target test` 的纯 zero-shot，也不是
`target dsss train abnormal -> target dsss test` 的单库监督训练，而是：

```text
source 学异常形态
target train normal 学目标域正常背景
target test 检验是否能在目标域发现异常
```

## 2. 当前架构

PromptAD 主干保留，新增一个 source-mask-supervised dense 分支：

```text
source/target spectrogram
  -> CLIP visual encoder
  -> multi-layer patch tokens
       ├─ Existing PromptAD branch
       │    -> text/image anomaly score
       │    -> target train normal gallery
       └─ New Dense Mask Branch
            -> raw dense map
            -> post dense map
            -> BCE + Dice mask supervision on public source masks
  -> final anomaly map / score
```

架构图：

```text
analysis_outputs/dense_mask_supervision_architecture/promptad_dense_mask_supervision_architecture.png
```

## 3. 已实施代码

新增/修改：

- `PromptAD/model.py`
  - 新增 `DenseMaskHead`
  - 新增 `dense_mask_branch` 开关
  - 新增 `calculate_dense_mask_logits`
  - 新增 `calculate_dense_mask_score`
  - `score_cached` 在开启 dense 分支时可使用 dense map 作为像素图和图像分数

- `train_rf_public_to_target_dense.py`
  - 当前唯一正式 dense 跨库入口
  - source 读取 `rf_public_pooled_smoke`
  - target 读取 `rf_target_test_pool`
  - 训练 dense 分支时只用 public source mask
  - 评估每个 target 类别和 JSR 前，先用对应 target train normal 建正常 gallery
  - dense loss = BCEWithLogits + Dice
  - source mask 下采样使用 adaptive max-pooling，避免小异常区域在 15x15 patch 网格中被 nearest 采样丢掉

- `datasets/rf_public_pooled_smoke.py`
  - public source pooled loader
  - train split: public normal gallery
  - test split: public normal + abnormal + mask，用作 dense 监督训练 query

- `datasets/rf_target_test_pool.py`
  - project target pooled loader
  - train split: target train normal only，用于 normal adaptation
  - test split: target test normal + abnormal + mask，用于最终评估

- `datasets/dataset.py`
  - GT 兼容解析
  - 旧 mask：yellow anomaly
  - 新 mask：white-on-black binary anomaly

旧 score-only/cross-only 入口已移除，正式实验只保留 dense 入口。

## 4. Smoke 验证

这个命令只验证代码链路，不作为正式效果：

```bash
python train_rf_public_to_target_dense.py \
  --method-tag dense_adapt_smoke \
  --cross-root-dir analysis_outputs/promptad_rf_cross_public_dense_adapt_smoke \
  --results-csv analysis_outputs/promptad_rf_cross_public_dense_adapt_smoke/results.csv \
  --source-noise-level m10db --eval-jsrs m10db \
  --pool-classes burst,dsss --target-classes dsss \
  --max-normal-per-class 2 --max-abnormal-per-class 2 \
  --Epoch 1 --batch-size 16 --gpu-id 0 --seed 111 \
  --prompt-mode rf --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only --vis False \
  --dense-lr 0.001 \
  --results-notes smoke_tiny_source_target_normal_adapt_not_for_quality
```

说明：

```text
source 每类只有 2 normal + 2 abnormal，不能用于论文结论。
target train normal 会被读取，但只用于建正常 gallery。
target test 才参与评估。
```

## 5. 正式实验建议

### 5.1 Universal dense source + target normal adaptation

```bash
python train_rf_public_to_target_dense.py \
  --method-tag dense_adapt_universal_A \
  --cross-root-dir analysis_outputs/promptad_rf_cross_public_dense_adapt \
  --results-csv analysis_outputs/promptad_rf_cross_public_dense_adapt/results.csv \
  --source-noise-level m10db --eval-jsrs m10db,m20db,m30db \
  --pool-classes burst,chirp,dsss --target-classes burst,chirp,dsss \
  --max-normal-per-class 300 --max-abnormal-per-class 300 \
  --Epoch 50 --batch-size 64 --gpu-id 0 --seed 111 \
  --prompt-mode rf --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only --vis False \
  --dense-lr 0.001 \
  --results-notes dense_public_source_supervision_target_train_normal_gallery
```

评估粒度：

```text
burst/chirp/dsss 分开报
m10db/m20db/m30db 分开报
avg 行表示同一类别跨 JSR 宏平均
class=all, jsr=avg 表示所有类别 × 所有 JSR 的总宏平均
```

checkpoint 保存规则：

```text
每个异常类型保存一个 best checkpoint:
  按该类型跨 JSR 平均 image ROC 选 best epoch

额外保存一个 all overall-best checkpoint:
  按所有类型 × 所有 JSR 的总平均 image ROC 选 best epoch
```

## 6. 当前不做的方向

以下方向已经移出正式主线：

- 使用 target abnormal/mask 的监督训练
- public source 训练后直接评估 target test 的纯 cross-only 路径
- 项目内信号类型互转的 pairwise 训练
- VAE reconstruction fusion
- 手工压制正常信号 / local variance / frequency z-score
- A-F score-fusion adapter 消融
