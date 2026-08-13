# OFDMA Few-Shot CLS 实验 Handoff

> **历史补充协议**：本文档记录的是旧官方随机划分实验，不是当前 OFDMA 主实验。
> 当前主实验使用 target-scene cold-start 协议，详见
> `docs/research/ofdma_target_scene_coldstart_protocol.md` 和
> `analysis_outputs/20260728_ofdma_patch_rank3_formal_test/README.md`。

## 1. 目标

在 OFDMA 干扰频谱公开数据集上，使用完全相同的 normal-only support 和官方测试集，比较：

- `promptad_text_vit_harmonic`：PromptAD 通用文本分数 + 原始 ViT patch memory。
- `ours_vit`：paired-TTA ViT memory。
- `ours_cnn`：CNN local memory。
- `confidence_gated_dual_visual`：实现兼容用的结果键；论文方法名为 Confidence Fusion，
  以 ViT 为主、CNN 按固定 support-derived confidence 提供非负补充。

只做图像级异常分类，不做分割，不训练异常样本，不按已知干扰类型选择模型或融合规则。
1/2/4-shot 都统一使用唯一固定的 support-only confidence fusion，不提供其他融合接口。

## 2. 数据与轴向

数据目录：

```text
/mnt/data/wangbei/data/ofdma-spectrum-anomalies-simulation/Example_Use/Dataset
```

原始 PNG 为 `height=1320, width=70`：

- 横轴：时间，共 70 个时间点。
- 纵轴：频率，共 1320 个子载波。
- 每个 scene 有 21 张 SU 频谱图，1-shot 表示一个完整正常 scene，即 21 张 support 图。

预处理严格复用数据集官方物理过程：先恢复 dB，在线性功率域每 12 个相邻子载波求和，再转回 dB，得到 `110x70`。之后保持宽高比缩放并复制边缘补齐到 `240x240`，不切时间、不切频率。

## 3. 固定协议

官方 normal-only 划分：

- 正常训练池：前 8,800 个正常 scene，只从这里选择 support。
- 正常验证集：接下来的 200 个正常 scene，本实验暂不用于选参。
- 正常测试集：最后 1,000 个正常 scene。
- 异常测试集：5 类干扰各最后 500 个 scene，共 2,500 个异常 scene。
- 正式测试图像：`(1000 + 5x500) x 21 = 73,500` 张。

标签只在全部分数生成完成后计算 AUROC、AUPRC 和 FPR@95TPR，不参与分数生成。

## 4. 两个方法

### PromptAD baseline

- 通用 `radio frequency spectrogram` prompt，`prompt_mode=legacy`。
- 当前正式 PromptAD checkpoint。
- 每个 support scene 的全部 ViT patch 建立原始完整 memory。
- 每个测试 patch 取正常 memory 的 1-NN 距离，图像分数取 anomaly map 最大值。
- 最终分数为 text score 与 ViT image score 的调和融合。
- 不使用我们方法的 coreset、top-k NN、paired TTA 或 CNN。

### 我们的方法

- ViT：固定 `layer1 + layer2 concat` patch 特征，Farthest-First Coreset 保留 50%，top-5 NN 距离取均值。
- Paired TTA：`identity / Gaussian blur / 水平左移4px / 水平右移4px`。support 和 test 使用同名视图，各自建立 memory，四张异常图按 max 融合。
- CNN：冻结 ImageNet ResNet18 `layer3`，完整正常 local memory，1-NN，最高 10% patch 距离均值作为图像分数；CNN 不使用 TTA 和 coreset。
- 融合：在当前测试单元内分别归一化 ViT/CNN，并计算 CNN 百分位排名；CNN 只有
  在排名较高且归一化证据强于 ViT 时才补充。
- 不保留旧的融合调参接口或融合调参命令行参数。

## 5. 执行命令

先检查数据协议和预处理，不加载模型：

```bash
mkdir -p analysis_outputs/20260715_ofdma_fewshot_comparison
python tools/eval_cls_ofdma_fewshot_comparison.py \
  --validate-only \
  --output-root analysis_outputs/20260715_ofdma_fewshot_comparison
```

最低成本端到端 smoke：

```bash
python tools/eval_cls_ofdma_fewshot_comparison.py \
  --shots 1 \
  --max-test-normal-scenes 1 \
  --max-test-scenes-per-jammer 1 \
  --max-sus-per-scene 1 \
  --batch-size 6 \
  --num-workers 0 \
  --gpu-id 2 \
  --output-root analysis_outputs/20260715_ofdma_fewshot_smoke
```

正式 1/2/4-shot 全量实验建议在 tmux 中运行：

```bash
mkdir -p analysis_outputs/20260715_ofdma_fewshot_comparison
python tools/eval_cls_ofdma_fewshot_comparison.py \
  --shots 1 2 4 \
  --gpu-id 2 \
  --batch-size 32 \
  --num-workers 8 \
  --output-root analysis_outputs/20260715_ofdma_fewshot_comparison \
  2>&1 | tee analysis_outputs/20260715_ofdma_fewshot_comparison/formal_run.log
```

脚本会根据 `--gpu-id 2` 设置可见卡，并在进程内使用 `cuda:0`。

## 6. 产出

```text
analysis_outputs/20260715_ofdma_fewshot_comparison/
├── protocol.json
├── support_1shot.json
├── support_2shot.json
├── support_4shot.json
├── results.csv
├── summary.json
├── formal_run.log
└── scores/
    ├── ofdma_1shot_image_scores.npz
    ├── ofdma_2shot_image_scores.npz
    └── ofdma_4shot_image_scores.npz
```

`results.csv` 同时提供：

- 5 种 jammer 各自指标。
- `macro_jammer` 五类宏平均。
- `overall` 全部异常统一评估。
- `image`、`scene_mean`、`scene_max` 三种层级。
- AUROC、AUPRC、FPR@95TPR 三个指标。

每完成一个 shot，脚本会立即更新 `results.csv` 和对应 NPZ；后续 shot 中断时，已完成结果仍然保留。

## 7. 已完成验证

- `py_compile` 通过。
- `--validate-only` 通过：原始 `1320x70`、聚合 `110x70`、模型输入 `240x240`。
- 自研预处理与官方 12 子载波聚合公式逐像素一致。
- 1 support scene + 6 test images 的端到端 smoke 通过。
- smoke 只验证程序正确连通，不作为正式性能结果。
