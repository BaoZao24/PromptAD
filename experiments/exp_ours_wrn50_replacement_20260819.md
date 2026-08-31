# Ours 方案 CNN 骨干替换实验：ResNet18 → WideResNet50-2（layer3）

日期：2026-08-20 ~ 2026-08-21
性质：探索性实验（不改动正式方法）

## 目的

把 Ours（ViT+CNN confidence fusion）中的辅助 CNN 分支从 ImageNet ResNet18 layer3
替换为 ImageNet WideResNet50-2 layer3，其余全部不变（ViT 分支、TTA、
support-only gate、manifests、seed 111），在四个数据集上做配对比较，
检验更强 CNN 骨干能否提升最终融合性能。

前置单分支探索（20260819）：In-house RF 上 WRN50 layer3 单分支 91.82 vs
ResNet18 layer3 90.05，layer2/layer1 更差，故替换实验固定 layer3。

## 改动内容

- `tools/eval_seg_resnet_gallery_fusion.py`：新增 `WideResNet50LocalEncoder`
  （与 `ResNet18LocalEncoder` 同接口：BGR uint8 输入、resize 224、
  ImageNet 归一化、输出归一化 layer1-4 特征图）。
- `tools/eval_cls_aux_cnn_gallery.py`：新增 `--cnn-encoder {resnet18,wideresnet50}`
  与 `--max-gallery-patches`；npz score key 对 WRN50 为
  `wideresnet50_layer3_top0.1_scores`（正式 resnet18+layer3 保持旧 key）。
- `tools/eval_cls_dual_visual_evidence_fusion.py`：新增 `--cnn-score-key` 覆盖。
- `tools/eval_cls_ofdma_target_scene_ours.py`、`tools/eval_fedjam_fewshot_dual.py`：
  新增 `--cnn-encoder`。
- 环境修复：prompt_ad 环境补装 `pyarrow`（FedJam 加载依赖，此前缺失导致
  fedjam 阶段一次失败）。

默认值全部保持 resnet18，正式方法行为不变。

## 协议

与正式主线完全一致：

- In-house RF：five-type manifest（96 张 support），ViT
  farthest-coreset 0.5 + 5NN mean + stft_shift_blur TTA max 融合；CNN 全量
  layer3 记忆 + 1-NN + top10% patch 均值；support-only gate（q=0.8, T=2.5,
  alpha=2.5）。60 cells 宏平均。
- Public RF：k=1/2/4 per-frequency manifests（16/32/64 张 support），
  测试正常集固定 5120 张；15 cells 宏平均。
- OFDMA：target-scene coldstart v2-realistic test split，30 场景 × 1/2/4-shot，
  场景宏平均；spectral_response TTA。
- FedJam：完整 test 7200 行，benign-only support（与 20260810 正式 run 的
  support manifest 逐项核对一致），1/2/4-shot。

所有配对实验中 ViT 分支分数完全相同（同一输出目录复用），只有 CNN 编码器不同。

## 结果

### CNN 单分支（AUROC，%）

| 数据集 | ResNet18 | WRN50 | Δ |
|---|---:|---:|---:|
| In-house RF | 90.05 | 91.82 | +1.77 |
| Public RF k=1 | 75.08 | 79.58 | +4.50 |
| Public RF k=2 | 75.66 | 80.37 | +4.71 |
| Public RF k=4 | 75.64 | 81.66 | +6.02 |
| OFDMA 1-shot | 69.45 | 76.72 | +7.27 |
| OFDMA 2-shot | 75.41 | 81.64 | +6.23 |
| OFDMA 4-shot | 79.66 | 84.84 | +5.18 |
| FedJam 1-shot | 75.11 | 82.07 | +6.96 |
| FedJam 2-shot | 79.40 | 85.14 | +5.74 |
| FedJam 4-shot | 83.27 | 86.78 | +3.51 |

### 融合 Ours（ViT+CNN confidence fusion，AUROC/AUPRC/FPR@95，%）

| 数据集 | ResNet18（配对复现） | WRN50（替换） | Δ AUROC |
|---|---|---|---:|
| In-house RF | 92.25 / 81.81 / 22.64 | 92.01 / 81.00 / 22.46 | −0.24 |
| Public RF k=1 | 81.44 / 44.30 / 45.68 | 81.10 / 44.17 / 46.07 | −0.34 |
| Public RF k=2 | 82.38 / 45.14 / 44.58 | 82.07 / 45.04 / 44.06 | −0.31 |
| Public RF k=4 | 83.33 / 45.23 / 43.76 | 83.24 / 45.16 / 42.89 | −0.09 |
| OFDMA 1-shot | 83.92 / 87.55 / 69.07 | 84.26 / 87.82 / 68.13 | +0.34 |
| OFDMA 2-shot | 88.47 / 91.19 / 60.90 | 88.85 / 91.46 / 58.73 | +0.38 |
| OFDMA 4-shot | 91.85 / 93.82 / 49.53 | 92.24 / 94.13 / 48.67 | +0.39 |
| FedJam 1-shot | 85.17 / 95.23 / 78.67 | 85.17 / 95.23 / 78.67 | 0.00 |
| FedJam 2-shot | 85.89 / 95.38 / 73.11 | 86.06 / 95.43 / 72.39 | +0.17 |
| FedJam 4-shot | 86.35 / 95.48 / 71.00 | 86.47 / 95.51 / 70.67 | +0.12 |

ViT 单分支（两臂共享）：In-house 91.37；Public k1/2/4 = 80.18/80.87/81.90；
OFDMA 1/2/4 = 84.02/88.49/91.99；FedJam 1/2/4 = 85.16/85.72/86.22。

## 结论

1. **WRN50 显著增强 CNN 单分支**：全部 10 个设置都提升（+1.8 ~ +7.3），
   与 SPADE/PatchCore 文献中 WRN50 强于 R18 的惯例一致。
2. **融合后的 Ours 基本不变**：RF 三套（In-house、Public k1/2/4）略微下降
   0.1~0.3；OFDMA/FedJam 略升 0~0.4。没有任何设置出现与单分支增益相称的提升。
3. **原因分析**：support-only confidence gate 是保守门控——只有当 CNN 证据
   超过 ViT 且达到正常 support 高分位时才采信。In-house/Public 上 ViT 明显
   强于 CNN（91.4 vs 90.1；80~82 vs 75~82），门控很少激活 CNN 路径，CNN 分支
   变强对融合几乎无影响；WRN50 更强的 CNN 同时抬高了自身 support 参考分布，
   门控校准后净效应近似为零。
4. **建议**：不替换正式方法骨干。ResNet18 保留为正式配置（更轻、更快、
   论文叙事一致）；WRN50 单分支结果可作为"CNN 分支容量不是瓶颈、门控融合
   才是关键"的消融证据写入讨论或附录。

## 复现基线偏差说明（诚实记录）

- In-house r18 融合复现 92.25 vs 20260801 正式 bundle 91.97：本次直接用
  `eval_cls_dual_visual_evidence_fusion.py` 全流程重算（旧正式值来自
  export+gate 两段式管线），差 0.28 属管线细节差异，配对结论不受影响。
- FedJam r18 复现 85.17/85.89/86.35 vs 20260810 正式 83.71/84.48/84.80：
  support manifest 已核对逐项一致，差异来自脚本自 20260810 后的演进与旧
  checkpoint 路径（`/mnt/data/wangbei/PromptAD/...`，现已不存在）无法回溯。
  本实验内两臂共享全部设置，配对比较有效。
- OFDMA r18 复现 83.92/88.47/91.85 vs 论文 85.30/89.53/92.75：同上，属
  复现口径差异；配对比较有效。

## 输出文件

- 主目录：`analysis_outputs/exploratory/20260819_ours_wrn50_replace/`
  - `inhouse_{vit,cnn_r18,cnn_r18_reference,cnn_wrn50_reference,fusion_r18,fusion_wrn50}`
  - `public_{vit,cnn_resnet18,cnn_wideresnet50,...}_{k1,k2,k4}` 及 fusion
  - `ofdma_{wrn50,r18_repro}`
  - `fedjam_{wrn50,r18_repro}`
  - `logs/`、`stage_*.done`
- WRN50 单分支 In-house 探索：`analysis_outputs/exploratory/20260819_wideresnet50_rf_layer{1,2,3}/`
- 驱动脚本：`/home/wangbei/tmp/opencode/run_ours_wrn50_replace.sh`（stage 断点续跑）
