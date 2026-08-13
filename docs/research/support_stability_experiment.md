# 审稿意见 2：support 抽样稳定性

## 问题

原来的少样本结果只固定使用一组 support。这样无法区分“方法本身的提升”和
“恰好抽到了一组有利的正常图”。因此不能只报告一次 support 下的 AUROC，必须
重复抽取 support，并报告重复之间的波动。

## 统一实验协议

- 测试图像和异常标签在所有重复中保持不变；只改变 normal support。
- 每个 seed 先生成 support manifest，再由 ViT、CNN 和 confidence fusion
  共同使用；融合阶段检查 support manifest 的 SHA-256。
- RF：self RF 按 scene、每个频段选一张 normal；public RF 先固定前 20 个完整
  `MeasRes_*` 记录作为 support 候选池，再按频段用 seeded hash 选一张 normal。
  候选池之外的完整记录永久作为 test，因此不同 seed 的 test 路径完全一致。
  public RF 的旧实现虽然接收 `--seed`，但实际每次选择同一批图，现已改为显式
  `--support-seed` 和共享 `--support-manifest`。
- OFDMA：同一 target scene 内按 seeded hash 选择完整 normal observation，保留
  1/2/4-shot 的 21-SU 结构；ViT 和 CNN 共用同一组 observation。
- 每个重复先做 cell-macro（RF）或 scene-macro（OFDMA），不把不同重复的测试
  图像拼在一起算 ROC。
- 汇总前对每个对应 score 文件校验有序的 test names、labels 和长度；RF 的
  ViT-only、CNN-only、融合以及 PromptAD score 使用同一 canonical cell key，避免
  方法之间悄悄使用不同测试单元。
- RF 的重复汇总同时包含 ViT-only、CNN-only、Direct OR、Ours 和同一 ViT evaluator
  输出的 PromptAD score；这些分数使用同一 support manifest 和固定 test split。
- OFDMA 额外保存 `support_manifest.json`，记录每个 scene、shot 的完整
  observation ID 和图像路径。

## 统计量

对每个方法和指标报告：

\[
\bar m=\frac{1}{R}\sum_{r=1}^{R}m_r,\qquad
s=\sqrt{\frac{1}{R-1}\sum_{r=1}^{R}(m_r-\bar m)^2},
\]

并给出重复级 95% CI。Ours 与 ViT-only、CNN-only 使用相同 seed 的配对差值；
`paired_deltas.csv` 同时给出 bootstrap CI 和 sign-permutation p 值。指标为
AUROC、AUPRC 和 FPR@95%TPR。

## 复现实验入口

```bash
python tools/run_support_stability_experiment.py \
  --output-root autoresearch/support-stability-rf-20-fixedtest \
  --seeds 111 222 333 444 555 666 777 888 999 1110 \
          1221 1332 1443 1554 1665 1776 1887 1998 2109 2220 \
  --datasets rf --gpu-id 0

python tools/summarize_support_stability.py \
  --run-root autoresearch/support-stability-rf-20-fixedtest \
  --output-root autoresearch/support-stability-rf-20-fixedtest/summary

python tools/plot_support_stability.py \
  --summary-csv autoresearch/support-stability-rf-20-fixedtest/summary/mean_std_ci95.csv \
  --output-root autoresearch/support-stability-rf-20-fixedtest/summary/figures

python tools/run_support_stability_experiment.py \
  --output-root autoresearch/support-stability-ofdma-5 \
  --seeds 111 222 333 444 555 \
  --datasets ofdma --gpu-id 0
```

输出包括每个 seed 的 manifest、分支分数、confidence-fusion 分数，以及：

- `per_replicate_metrics.csv`：每个重复的宏平均结果；
- `rf_signal_mean_std_ci95.csv`：按 RF 异常类型汇总的均值、标准差和 95% CI；
- `mean_std_ci95.csv`：均值、标准差和 95% CI；
- `paired_deltas.csv`：Ours 相对 ViT-only/CNN-only 的配对差值和 bootstrap CI；
- `paired_deltas.csv` 也包含 Ours 相对 PromptAD 的 RF 配对差值；
- `ofdma_scene_bootstrap_ci95.csv` 与 `ofdma_paired_scene_bootstrap.csv`：以 30 个
  target scene 为重采样单位的 OFDMA 置信区间和配对差值，避免把观测图像直接混合。
- `ofdma_jammer_mean_std_ci95.csv`：按 jammer 类型和 4-shot 汇总重复 support 的
  均值、标准差和 95% CI，可直接生成带误差棒的类型分解图。

## 需要在论文中说明的边界

OFDMA 当前数据生成协议每个 test scene 只有 4 个 normal support observation，
因此 4-shot 在“不放回抽样”下天然只有一个组合；1/2-shot 才能观察 support
抽样方差。若要对 4-shot 也做真正的多组合重复，需要扩大每个 scene 的 normal
support pool，而不能把同一 observation 重复计作新的独立样本。

## 当前状态

代码和哈希一致性检查已完成。汇总器现在同时检查测试名称、标签序列和长度；RF 正式实验已完成 20 个
test 路径哈希完全相同、support 路径哈希两两不同；self RF 的每个对应 cell test
hash 也在汇总时通过一致性校验。RF 汇总位于
`autoresearch/support-stability-rf-20-fixedtest/summary/`。

OFDMA 的 5 个 seed 已按上述入口全部完成，并额外保存 scene-level bootstrap 统计；
汇总位于 `autoresearch/support-stability-ofdma-5/summary/`。OFDMA score 文件使用
`target_scene_ids + observation_ids` 组成 canonical observation key，因此即使不含 RF
的 `names` 字段，也能对不同方法和重复执行同样的 scene/observation/label 一致性校验。
论文中的 OFDMA 稳定性表和误差棒均来自该汇总目录，不再使用单次或部分重复结果。
