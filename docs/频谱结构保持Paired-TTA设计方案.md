# 频谱结构保持 Paired-TTA 设计方案

## 先说结论

我们已经实施了一个专门面向频谱图的四视图 TTA 候选：

| 数据 | 四个视图 |
|---|---|
| RF | 原图、时间轴上移 4 px、时间轴下移 4 px、背景噪声底抖动 |
| OFDMA | 原图、时间轴左移 4 px、时间轴右移 4 px、背景噪声底抖动 |

代码入口为 `rf_spectral_time_background_v1` 和
`ofdma_spectral_time_background_v1`。它们是显式启用的探索性候选；现有正式配置
`stft_shift_blur` 和 `ofdma_time_shift_blur` 保持不变。

## 1. 为什么要这样设计

频谱图不是普通图片：横轴和纵轴有固定的物理含义，通常分别表示时间和频率。因此，普通
图像中的翻转、旋转或随意裁剪可能会破坏通信信号结构。我们的 TTA 只模拟两类更合理的
变化：

1. **时间位置变化**：STFT 窗口起点、触发时刻或采集对齐略有变化时，同一个合法信号可能
   在时间轴上前后移动。双向时间平移让 memory 不把这种对齐差异误判为异常。
2. **背景噪声底变化**：接收机噪声、环境底噪和增益状态变化会使低功率背景出现空间上缓慢
   的起伏，但通常不会直接改变亮的信号结构。因此只对低功率背景增加平滑、局部且零均值的
   小扰动，让模型学习“噪声底变化仍可能是正常的”。

具体实现中，背景区域由每张图灰度值的 65% 分位数估计；噪声场由确定性平滑场和少量平滑
颗粒组成，默认强度为 3 个灰度级。亮线、脉冲、能量块等高功率信号不直接加噪，正负局部
变化在同一张图内同时出现，避免把整张图变亮后被 max 融合偏好。

这四个视图的目的不是制造异常，而是覆盖频谱采集中的正常变化。原图始终保留，避免增强
把正常结构改得过多。

## 2. Paired-TTA 是什么

Paired 的意思是：support 和 query 使用同一组变换，再进行特征比较。

```text
正常 support ──> 原图/时间左或上/时间右或下/背景噪声底 ──> 建立正常 memory
待测 query  ──> 原图/时间左或上/时间右或下/背景噪声底 ──> 查询各视图 memory
                                                  ↓
                                      融合各视图 ViT 异常分数
                                                  ↓
                                      再与 CNN layer3 分数融合
```

设四个视图为

$$
\mathcal{T}(x)=\{x, T_{-}(x), T_{+}(x), B_{\mathrm{jitter}}(x)\}.
$$

对每个视图独立建立或查询 ViT normal memory，得到分数
$s_V^t(x)$。当前候选使用固定的逐视图最大融合：

$$
s_V(x)=\max_{t\in\mathcal{T}}s_V^t(x).
$$

这个 max 只是在不同合理视图中选择最能反映异常的证据，不代表使用异常标签选择视图。随后
仍按原方案与 CNN ResNet18 `layer3` 的局部分数进行 confidence fusion；CNN 分支不新增
这组 TTA，以便单独观察频谱 TTA 对 ViT 分支的影响。

TTA 只改变输入视图，不更新模型参数，不读取异常标签，也不使用同一测试批次的统计量。

## 3. 当前代码状态

| 方向 | 代码状态 | 当前定位 |
|---|---|---|
| RF 时间双向平移 | 已实现 | 新四视图候选的一部分；RF 旧正式 TTA 仍保留 |
| OFDMA 时间双向平移 | 已实现 | 新四视图候选的一部分；方向按 OFDMA 坐标定义 |
| 背景噪声底抖动 | 已实现 | 新四视图的频谱专用设计，默认强度 3 |
| 频率方向平移/平滑 | 已实现 | 保留为历史候选，不放入当前四视图 |
| 频带宽度变化 | 已移出 | 容易把真正宽带异常变成“正常变化” |
| 功率残差 | 独立实现 | 是功率异常分支，不属于 TTA |

历史候选仍可通过 `utils/spectral_tta.py` 调用，但不能把历史候选的结果混入新四视图
结论。当前四视图的显式命令示例：

```bash
python tools/eval_cls_vit_patchcore_gallery.py \
  --paired-tta rf_spectral_time_background_v1 \
  --paired-tta-background-noise-strength 3
```

OFDMA 正式 target-scene 入口使用同样的候选时，必须显式添加：

```bash
python tools/eval_cls_ofdma_fewshot_comparison.py \
  --paired-tta ofdma_spectral_time_background_v1 \
  --paired-tta-background-noise-strength 3
```

正式 target-scene 入口仍保持冻结配置；上面命令用于独立的探索性 bundle 对照。

正式入口默认值没有被修改，实验协议会记录实际使用的 bundle 和视图列表。

## 4. 已完成的受控实验

实验只使用单 GPU、`num-workers=0`，并保留旧版配置作为对照。

| 数据/设置 | 旧 TTA | 新四视图候选 | 结果 |
|---|---:|---:|---|
| In-house RF burst，3 个 cell，ViT patch max AUROC | 99.48 | 95.41（强度 3） | 当前 smoke 未超过旧版 |
| In-house RF，同一设置，top-0.05 AUROC | 97.53 | 94.75（强度 3） | 当前 smoke 未超过旧版 |
| Public RF burst，3 个 cell，ViT patch max AUROC | 87.61 | 87.65 | 小幅提升，仍需完整验证 |
| Public RF burst，同一设置，top-0.05 AUROC | 81.77 | 81.89 | 小幅提升，仍需完整验证 |
| OFDMA `test_000`、1-shot、小规模 smoke，overall AUROC | 100.00 | 95.00 | 当前 smoke 下降 |

RF 噪声强度 2/3/4 的 max AUROC 分别为 94.89/95.41/95.38，说明仅调节噪声幅度不能
解决 RF smoke 中的下降。因此，新四视图目前应作为**已实施、理由明确但尚未升级为正式主线**
的探索设计；正式论文主表仍使用冻结的原 TTA 配置，除非后续 validation 实验显示跨数据集
稳定提升。

## 5. 后续实验原则

- 先在 validation scene 上固定视图和噪声强度，再锁定 test protocol；不能用 test 最优值
  反向挑选 TTA。
- 报告原图、时间平移、背景抖动和组合方案的独立消融，但不把 TTA 变体当作外部独立方法。
- 逐个数据集单独画图，保持视图顺序、颜色和标注风格统一。
- RF、OFDMA 的时间轴方向不同，不能直接复用同一方向名称；必须以数据集坐标定义为准。
- 如果新方案不能在多个数据集上稳定改善 AUROC、AUPRC 或 FPR@95%TPR，就只作为方法设计
  分析和补充实验，不写成已验证的主性能贡献。

## 6. 一句话创新表述

> 针对频谱图时间轴和频率轴具有固定物理语义、且接收噪声底会随环境变化的特点，设计轴语义
> 保持的 Paired-TTA：用双向时间对齐扰动覆盖采集窗口变化，用仅作用于低功率背景的零均值
> 噪声底抖动覆盖接收背景变化；support 与 query 同步处理，使这些正常频谱变化不会被误判
> 为异常。
