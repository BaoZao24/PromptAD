# UniVAD 全部实验结果独立记录（2026-08-16）

本文档单独记录当前已经完成的 UniVAD 相关实验，不替代论文主表，也不把未完成的 OFDMA 多场景结果写成正式结论。

## 1. 统一实验口径

### 1.1 方法名称

由于原始 UniVAD 的 C³、CAPM 和 GECM 分支需要物体或组件分割掩码，而频谱图没有对应的工业物体分割标注，当前实验只保留其整体图像纹理匹配路径。因此本文统一使用：

`UniVAD-Texture-adapted (DINOv2-G)`

这表示“基于 UniVAD 思路的频谱图纹理路径适配版”，不是完整官方 UniVAD。实验中使用 CLIP-L/14 和官方 DINOv2-G/14 特征；只用正常 support 建立记忆，不训练目标域模型，不用异常样本建库。

### 1.2 指标和比较对象

- AUROC、AUPRC：越高越好；
- FPR@95%TPR：越低越好；
- `Ours`：各数据集当前正式协议下的现有主线结果，采用对应 support/test 划分，`TTA=none`；
- 所有正式结果均使用完整测试集或完整测试单元，不使用早期的 8+8 小样本测试；
- 表格中的宏平均按测试单元或场景等权计算，不把不同数据集的样本直接混合。

### 1.3 当前结果覆盖范围

| 数据集 | 结果范围 | 状态 | 是否可作为正式结论 |
|---|---|---|---|
| In-house RF | 60 个 `signal × scene × JSR` 单元 | 已完成 | 是 |
| Public RF | 15 个单元 × `k=1/2/4-per-frequency`，共 45 个 cell-k 结果 | 已完成 | 是 |
| FedJam | 完整 7,200 条 test，1/2/4-shot | 已完成 | 是 |
| OFDMA | `test_000`，200 个 observation，1/2/4-shot | 已完成单场景验证 | 暂不能代表 30 场景宏平均 |
| DINOv2-B/G 3 单元预筛 | 3 个 In-house RF 单元 | 已完成 | 仅作适配流程预筛 |
| smoke test | 极小测试子集 | 已完成 | 否 |

## 1.4 UniVAD 使用的 CLIP/DINO 特征与融合方式

这里需要把 UniVAD 的特征和当前 Ours 区分开。当前实验中的 UniVAD 适配版使用的是：

| 特征来源 | 具体内容 | 用途 |
|---|---|---|
| CLIP-L/14-336 图像全局特征 | 整张频谱图的 global/CLS 向量 | 计算整图与正常 support 的距离 |
| CLIP-L/14-336 patch 特征 | `out_layers=[6,12,18,24]` 输出的多层 patch token，经 decoder 后使用 | 生成局部 CLIP 异常图 |
| DINOv2-G/14 patch 特征 | `x_norm_patchtokens`，每个频谱图的局部 patch 向量 | 生成局部 DINO 异常图 |
| CLIP 文本特征 | UniVAD 固定的 `object` 正常/异常 prompt 集合 | 生成文本引导的异常图 |

因此，UniVAD 这里的“CLIP 特征”包括两种图像侧特征：整图向量和 patch 向量；另外还有一组文本侧的 CLIP prompt 特征。它不是把 CLIP、DINO 和文本向量在通道维度直接 concat，而是先分别计算异常分数，再在分数层融合。

当前 Ours 不一样：使用 PromptAD 的 CLIP ViT-B/16-plus-240，局部分支使用项目定义的 `visual_features[2] + visual_features[3]`，不使用 DINO；正式 `confidence_gated_dual_visual` 也没有把 PromptAD 的文本分数纳入最终分数，CNN 辅助分支使用 ResNet18 layer3。

### UniVAD 的正常 memory

每张正常 support 图像分别建立三类 memory：

- 一条 CLIP global memory；
- 一组 CLIP patch memory；
- 一组 DINO patch memory。

当前适配器保留 support 的全部 patch，不使用 coreset。对每个待测 patch，都会在正常 patch memory 中寻找最相似的正常 patch，即 1-NN 匹配。

### UniVAD 的分数计算

为避免公式显示异常，下面使用普通文字表达。`max_cos` 表示与正常 memory 中所有向量的最大余弦相似度。

```text
global_score
    = 1 - max_cos(query_global, normal_CLIP_global_memory)

CLIP_patch_map
    = average over selected CLIP layers(
          1 - max_cos(each_query_patch, normal_CLIP_patch_memory)
      )

DINO_patch_map
    = 1 - max_cos(each_query_patch, normal_DINO_patch_memory)

text_patch_map
    = CLIP patch 与 abnormal prompt 的相似度
      和 CLIP patch 与 normal prompt 的相似度
      经过 softmax 后得到的 abnormal probability
```

先把三个局部异常图做固定等权平均：

```text
local_map
    = (CLIP_patch_map + DINO_patch_map + text_patch_map) / 3
```

最后取局部图的最大值，再加上整图 global 分数：

```text
final_score
    = max(local_map) + global_score
```

### 公式的直观含义

- `CLIP_patch_map`：检查每个局部频谱区域能不能在正常 support 中找到相似区域。如果某个局部区域找不到正常对应物，它的异常分数就会升高。
- `DINO_patch_map`：使用另一套视觉特征重新检查局部区域。它不完全依赖 CLIP，可以补充 CLIP 没有识别出来的纹理或结构变化。
- `text_patch_map`：利用 CLIP 的正常/异常文字描述给局部区域提供一个异常方向。例如，它判断某个 patch 更接近“normal object”还是“abnormal object”。它不是从测试集学习出来的，而是 UniVAD 固定的文本 prompt 先验。
- `local_map`：把三种局部证据等权平均。这里不是把三个特征向量拼起来，而是把三个异常图的分数放在一起平均，因此任何一个局部证据都可以对最终局部判断产生影响。
- `max(local_map)`：取整张图中最异常的区域。这样只要有一个明显的异常频谱区域，就可以提高图像级异常分数，适合局部突发干扰。
- `global_score`：检查整张频谱图的整体外观是否偏离正常 support。如果异常导致整体功率、背景纹理或宽带结构发生变化，即使局部异常图不够突出，global 分数也会把最终分数抬高。
- 最后的加法表示“局部异常 + 整体异常”。UniVAD 没有让 global 分支去抵消局部分数，而是把它作为额外证据直接加上去。

例如，假设某张待测图的三个局部分数分别为 `0.6`、`0.8`、`0.7`，则：

```text
local_map_score = (0.6 + 0.8 + 0.7) / 3 = 0.7
```

如果整图与正常 support 的 global 距离为 `0.4`，最终分数就是：

```text
final_score = 0.7 + 0.4 = 1.1
```

这个例子体现了两种互补情况：局部突发异常主要依赖 `max(local_map)`，而整体功率或背景发生变化时，`global_score` 会提供额外增益。

需要注意，这个公式是当前 UniVAD 纹理适配器的固定分数融合方式：没有可学习权重、没有门控，也没有把四类特征直接 concat。三个局部分支固定各占三分之一，global 分支固定以系数 `1` 加入。因此它更像“多种异常证据的固定集成”，而不是一个通过目标域训练得到的融合网络。

因此，UniVAD 的融合关系是：

```text
CLIP patch evidence ─┐
DINO patch evidence ─┼─> 三者等权平均 ─> 局部最大异常分数 ─┐
CLIP text evidence ──┘                                      ├─> final_score
CLIP global evidence ───────────────────────────────────────┘
```

它没有使用可学习融合层，也没有使用我们当前的 CNN 置信度门控；三个局部分支权重固定为 `1/3`，global 分数以固定系数 `1` 加到局部分数上。该公式是本次 `UniVAD-Texture-adapted` 适配器实际采用的实现口径。

## 2. 正式全量结果

### 2.1 In-house RF：60 单元宏平均

每个单元使用 1 张正常 support，测试集使用该单元的完整正常/异常测试样本。Ours 使用当前 support-only 主线，未使用 TTA。

| 方法 | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ |
|---|---:|---:|---:|
| UniVAD-Texture-adapted (DINOv2-G) | 88.47 | 76.75 | 30.17 |
| **Ours** | **91.41** | **80.18** | **23.58** |
| UniVAD − Ours | -2.94 | -3.43 | +6.58 |

结论：在完整 In-house RF 协议上，当前 Ours 的三项指标均优于 UniVAD 纹理适配版。

原始结果：[`inhouse_full_60cells.json`](../autoresearch/univad-official-260816/inhouse_full_60cells.json)。

### 2.2 Public RF：15 单元 × k-per-frequency

Public RF 使用固定的 5,120 张正常 test 图和全部异常 test 图；`k=1/2/4` 三个 support memory 严格嵌套。每个 k 均在 15 个单元上做宏平均。

| Support | UniVAD AUROC | Ours AUROC | UniVAD AUPRC | Ours AUPRC | UniVAD FPR95 | Ours FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| k=1 | 81.04 | **81.41** | 37.83 | **45.08** | **43.46** | 46.31 |
| k=2 | 81.66 | **83.55** | 40.38 | **45.30** | 44.00 | **43.98** |
| k=4 | 82.33 | **84.18** | 41.18 | **45.73** | 43.91 | **43.50** |

结论：UniVAD 随 support 数量增加而逐步提升，但三档的 AUROC 和 AUPRC 都低于 Ours；FPR@95%TPR 只有 k=1 低于 Ours，k=2/4 与 Ours 接近。

原始结果：[`public_full_k1k2k4.json`](../autoresearch/univad-official-260816/public_full_k1k2k4.json)。

### 2.3 FedJam：完整 7,200 条 test

FedJam 只使用 spectrogram 图像。support 从 train 中的 benign 样本抽取，test 包含 1,800 条 benign 和 5,400 条异常记录；KPI 时序不参与评分。

| Support | UniVAD AUROC | Ours AUROC | UniVAD AUPRC | Ours AUPRC | UniVAD FPR95 | Ours FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| 1-shot | **93.24** | 84.24 | **98.02** | 94.92 | **58.44** | 81.44 |
| 2-shot | **93.69** | 84.53 | **98.12** | 94.91 | **49.56** | 76.56 |
| 4-shot | **94.06** | 85.05 | **98.21** | 95.00 | **46.22** | 72.22 |

结论：FedJam 是当前 UniVAD 纹理适配版优势最明显的数据集，三档 support 下三项指标均优于 Ours。

原始结果：[`fedjam_full_1shot2shot4shot.json`](../autoresearch/univad-official-260816/fedjam_full_1shot2shot4shot.json)。

### 2.4 OFDMA：`test_000` 单场景完整验证

本组完成 `test_000` 的全部 200 个 observation，每个 observation 包含 21 张 SU 频谱图。先对 SU 图逐张评分，再取 observation 内 21 个分数的最大值。当前只完成了一个 target scene，不能写成 OFDMA 30 场景宏平均。

| Support | UniVAD AUROC | Ours AUROC | UniVAD AUPRC | Ours AUPRC | UniVAD FPR95 | Ours FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| 1-shot | **80.30** | 79.86 | **84.54** | 84.53 | **78.00** | 84.00 |
| 2-shot | 83.73 | **89.24** | 86.80 | **91.43** | 65.00 | **51.00** |
| 4-shot | 85.85 | **89.03** | 88.12 | **91.70** | 62.00 | 62.00 |

结论：在 `test_000` 上，1-shot 时两种方法接近；2/4-shot 时 Ours 的 AUROC、AUPRC 更高，2-shot 的 FPR@95%TPR 也更低，4-shot 两者相同。其余 29 个场景尚未完成，因此这里只作单场景验证。

原始结果：[`ofdma_test000_full.json`](../autoresearch/univad-official-260816/ofdma_test000_full.json)。

## 3. 早期 3 单元预筛结果

这两组实验用于检查 DINOv2 backbone 和频谱图适配流程，不替代 60 单元正式结果。Ours 数值已经统一为当前 support-only、`TTA=none` 的对应单元结果。

### 3.1 DINOv2-B 预筛

| 信号 | 场景 | 强度 | UniVAD-Texture-B AUROC | Ours AUROC | UniVAD − Ours |
|---|---|---:|---:|---:|---:|
| burst | WeaponMuseum | m30 dB | 67.01 | 87.59 | -20.581 |
| chirp | Playground | m20 dB | 93.22 | 99.70 | -6.481 |
| DSSS | TimeSquare | m20 dB | 93.31 | 94.60 | -1.293 |
| **三单元宏平均** | — | — | **84.51** | **93.96** | **-9.452** |

该组只保留了 AUROC 预筛结果，不能与后面的三指标正式全量表直接合并排名。

### 3.2 官方 DINOv2-G 预筛

| 信号 | 场景 | 强度 | UniVAD-Texture-G AUROC | Ours AUROC | UniVAD − Ours |
|---|---|---:|---:|---:|---:|
| burst | WeaponMuseum | m30 dB | 91.58 | 87.59 | +3.997 |
| chirp | Playground | m20 dB | 99.73 | 99.70 | +0.035 |
| DSSS | TimeSquare | m20 dB | 95.32 | 94.60 | +0.712 |
| **三单元宏平均** | — | — | **95.54** | **93.96** | **+1.581** |

该组 UniVAD 的 AUPRC 依次为 78.38%、99.65%、87.87%，FPR@95%TPR 依次为 45.28%、0.00%、11.29%；这些数值仍只用于预筛，正式结论以 DINOv2-G 的 60 单元全量结果为准。

## 4. Smoke test 结果

下面结果用于确认数据读取、support 构造、特征编码和指标计算流程，测试样本非常少，不能用于方法比较，也不能写入正式主表。

| 数据集 | 测试规模 | Support | AUROC | AUPRC | FPR@95%TPR | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Public RF | 1 个单元，4 normal + 4 abnormal | k=1/2/4 | 100.00 | 100.00 | 0.00 | 三个 k 均相同；仅流程检查 |
| FedJam | 2 normal + 6 abnormal | 1/2/4-shot | 100.00 | 100.00 | 0.00 | 三个 shot 均相同；仅流程检查 |
| OFDMA `test_000` | 6 observations | 1/2/4-shot | 100.00 | 100.00 | 0.00 | 每类异常取极少量样本；仅流程检查 |

原始 smoke 文件：

- Public RF：[`public_smoke.json`](../autoresearch/univad-official-260816/public_smoke.json)、[`public_smoke_multi.json`](../autoresearch/univad-official-260816/public_smoke_multi.json)、[`public_smoke_reuse.json`](../autoresearch/univad-official-260816/public_smoke_reuse.json)；
- FedJam：[`fedjam_smoke.json`](../autoresearch/univad-official-260816/fedjam_smoke.json)；
- OFDMA：[`ofdma_test000_smoke.json`](../autoresearch/univad-official-260816/ofdma_test000_smoke.json)。

## 5. 统一结论

1. In-house RF 和 Public RF 的完整协议上，当前 Ours 整体优于 UniVAD 纹理适配版。
2. FedJam 上，UniVAD 纹理适配版明显优于当前 Ours，是目前最有利于 UniVAD 的数据集。
3. OFDMA 当前只完成 `test_000`，结果为混合表现，不能外推到整个 OFDMA 数据集。
4. UniVAD 的结果必须写作 `UniVAD-Texture-adapted (DINOv2-G)`，不能简称为完整官方 UniVAD；C³、CAPM、GECM 因缺少频谱图物体/组件掩码没有纳入。
5. 当前可以报告的是“不同频谱数据集上的整体纹理匹配适配结果”，而不是“完整 UniVAD 在频谱异常检测上已经复现”。

## 6. 复现入口

### 6.1 实验脚本

- In-house RF：[`tools/eval_univad_rf_fewshot.py`](../tools/eval_univad_rf_fewshot.py)；
- DINOv2-B 适配预筛：[`tools/eval_univad_rf_texture_b_adapted.py`](../tools/eval_univad_rf_texture_b_adapted.py)；
- Public RF：[`tools/eval_univad_public_rf.py`](../tools/eval_univad_public_rf.py)；
- FedJam：[`tools/eval_univad_fedjam.py`](../tools/eval_univad_fedjam.py)；
- OFDMA：[`tools/eval_univad_ofdma.py`](../tools/eval_univad_ofdma.py)。

### 6.2 原始代码和论文

- UniVAD 源码：[`references/UniVAD/`](../references/UniVAD/)；
- UniVAD 论文 PDF：[`references/papers/univad_cvpr_2025.pdf`](../references/papers/univad_cvpr_2025.pdf)；
- 全部本次结果 JSON：[`autoresearch/univad-official-260816/`](../autoresearch/univad-official-260816/)；
- 较早的 RF 适配说明：[`exp_univad_rf_comparison_20260815.md`](exp_univad_rf_comparison_20260815.md)。

## 7. 待补工作

OFDMA 还需要补跑其余 29 个场景的 1/2/4-shot，并按场景做宏平均。完成前，本文中的 OFDMA 结果只能作为 `test_000` 单场景验证，不能与 In-house RF、Public RF、FedJam 的全量结果等量齐观。
