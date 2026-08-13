# 传统通信基线与频谱专用 TTA 实验协议

> **文献复核说明（2026-08-11）**：本文件前面的 ED、功率谱熵、谱平坦度、谱峭度和
> CA-CFAR 表格现在只表示频谱图域的诊断性适配，不再表示对论文通信异常检测算法的完整复现。
> 已重新核验的频谱异常论文方法及其主表筛选规则见
> [`docs/research/频谱异常检测文献复核与基线筛选.md`](../research/频谱异常检测文献复核与基线筛选.md)。
> 新主比较统一保留 ED、CA-CFAR、SCSE、KLD-Ref、IAD-PER、SAIFE、UDMA-ResNet18、
> PatchCore、WinCLIP 和 Ours；其中频谱方法均明确标注为当前 PNG 频谱图输入下的适配或
> 论文驱动重实现，不冒充原论文的逐位复现。ICA-Frozen 等已完成方法移入补充记录。

本文的主比较统一使用：ED、CA-CFAR、SCSE、KLD-Ref、IAD-PER、SAIFE、UDMA-ResNet18、
PatchCore、WinCLIP 以及本文的 ViT+CNN Confidence Fusion。ViT-only、CNN-only 只在独立
消融表中出现；外部异常检测基线只在已经完成同协议运行的数据集上报告。其中 SAIFE 是通信
频谱深度生成式方法，不是通用视觉方法。旧提示词基线结果只保留在历史运行目录中，不进入
论文实验表、结论或比较性文字。

## 1. 传统通信基线的边界

当前 In-house RF、Public RF、OFDMA target-scene 和 FedJam 数据给出的输入均为 PNG 频谱图。
其中 In-house RF 的正常背景来自本项目自测/自采集的 RF 记录，异常样本由本项目注入合成
干扰生成；它不是外部公开数据集。其余数据集的出处和构造边界见论文实验部分第 4.7 节。
不是复数 IQ、时间序列或多天线协方差快拍。因此下面的定量结果统一命名为
“spectrogram-domain adaptation”，表示把经典统计量应用到频谱图上；不能表述为完整复现
需要 IQ 输入的通信检测器。

| 方法 | 本项目中的固定实现 | 论文出处 |
|---|---|---|
| ED | 频谱图强度平方的均值（support-only robust z-score） | [1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503) Urkowitz (1967), [DOI](https://doi.org/10.1109/PROC.1967.5573) |
| Power-spectrum entropy | 沿时间聚合得到频率功率剖面，再计算归一化 Shannon entropy；低熵为异常证据 | [2](https://jcs.iie.ac.cn/xxaqxb/ch/reader/view_abstract.aspx?doi=10.19363%2Fj.cnki.cn10-1380%2Ftn.2023.08.21) 洪泽坤等（录用稿）, [DOI](https://doi.org/10.19363/J.cnki.cn10-1380/tn.2023.08.21) |
| Spectral flatness | 频率功率剖面的 geometric mean / arithmetic mean；低平坦度为异常证据 | [3](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2016.4712) Gurugopinath (2017), [DOI](https://doi.org/10.1049/el.2016.4712) |
| Spectral kurtosis | 每个频率 bin 沿时间计算 excess kurtosis，取 95th percentile | [4](https://www.sciencedirect.com/science/article/pii/S0888327004001517/pdf) Antoni (2006), [DOI](https://doi.org/10.1016/j.ymssp.2004.09.001) |
| CA-CFAR | 频率功率剖面上的 guard/train cells 局部均值和标准差，自适应计算最大正残差 | [5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829) Rohling (1983), [DOI](https://doi.org/10.1109/TAES.1983.309350) |
| SCSE（Support-Calibrated Spectral Ensemble） | 五个统计量分别用正常 support 的 median/IQR/MAD 校准，取最大正异常证据；没有异常标签调权 | 本项目派生基线，组件出处为 [1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503)–[5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829) |
| KLD-Ref | 正常 support 拟合冻结灰度直方图，查询图计算 $D(P\|Q)$；采用 Afgani 方法的 spectrogram-domain adaptation，不使用查询样本更新参考 | [16](../references/papers/spectrum_domain/17_afgani_2010_information_theoretic.pdf) Afgani et al. (2010) |
| ICA-Frozen | 正常 support 拟合事件信息量，连续高信息事件形成图像分数；冻结测试阶段直方图 | [16](../references/papers/spectrum_domain/17_afgani_2010_information_theoretic.pdf) Afgani et al. (2010) |
| IAD-PER | 正常 support 训练 VAE，按论文公开 PER 公式计算背景/信号百分位重建分数 | [6](../references/papers/06_ism_2022_vae_anomaly.pdf) Tian et al. (2022) |
| UDMA-ResNet18 | 教师 CNN、AE 学生、MemAE 学生及三路差异；预训练参考网络因原文未指定而固定为 ImageNet ResNet18 | [17](<../references/UDMA/Qi 等 - 2024 - Unsupervised Spectrum Anomaly Detection With Distillation and Memory Enhanced Autoencoders.pdf>) Qi et al. (2024) |

Energy detection 的经典定义可追溯到 Urkowitz；CFAR 的局部自适应阈值来自 Rohling。谱峭度用于非平稳/非高斯成分刻画，谱平坦度用于频谱感知，分别对应 Antoni 与 Gurugopinath 的工作。

五个单项统计量是 SCSE 的组成部分，SCSE 是四个数据集共同使用的非学习通信基线；它们与
KLD-Ref、IAD-PER、SAIFE、UDMA-ResNet18 及视觉基线使用相同的 normal-only、support-only
和最终指标口径。ViT-only、CNN-only 仅在独立消融表中出现。外部深度/视觉异常检测方法不
属于本节的传统统计通信基线；其数据集覆盖情况见 [`论文实验部分.md`](../paper/论文实验部分.md)
的统一方法覆盖表。

本项目采用的完整参考文献如下：

1. H. Urkowitz, “Energy Detection of Unknown Deterministic Signals,” *Proceedings of the IEEE*, vol. 55, no. 4, pp. 523–531, 1967. [DOI](https://doi.org/10.1109/PROC.1967.5573) [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503)
2. 洪泽坤、徐艳云、黄伟庆、李婷婷、于超，《基于功率谱熵的隐蔽通信信号检测技术研究》，《信息安全学报》，录用稿。 [DOI](https://doi.org/10.19363/J.cnki.cn10-1380/tn.2023.08.21) [PDF/官网页面](https://jcs.iie.ac.cn/xxaqxb/ch/reader/view_abstract.aspx?doi=10.19363%2Fj.cnki.cn10-1380%2Ftn.2023.08.21)
3. S. Gurugopinath, “Robust Spectrum Sensing Based on Spectral Flatness Measure,” *Electronics Letters*, vol. 53, no. 13, pp. 890–892, 2017. [DOI](https://doi.org/10.1049/el.2016.4712) [PDF](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2016.4712)
4. J. Antoni, “The Spectral Kurtosis: A Useful Tool for Characterising Non-Stationary Signals,” *Mechanical Systems and Signal Processing*, vol. 20, no. 2, pp. 282–307, 2006. [DOI](https://doi.org/10.1016/j.ymssp.2004.09.001) [PDF](https://www.sciencedirect.com/science/article/pii/S0888327004001517/pdf)
5. H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target Situations,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-19, no. 4, pp. 608–621, 1983. [DOI](https://doi.org/10.1109/TAES.1983.309350) [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829)

以下方法暂不放入定量主表：cyclostationary detection（Gardner, 1991, [DOI](https://doi.org/10.1109/79.81007)）和 eigenvalue-based sensing（Dikmese & Renfors, 2012, [DOI](https://doi.org/10.4108/icst.crowncom.2012.248467)）。它们需要循环谱相关、时间快拍、协方差矩阵或多传感器输入；仅凭当前 PNG 频谱图做“代理实现”会改变算法定义，故列为后续拿到 IQ/原始时序后的扩展实验。

## 2. 公平评估协议

- 所有统计量只使用已确认正常 support 估计 median、IQR 和 MAD；异常测试图像和测试标签只在最后计算 AUROC、AUPRC、FPR@95%TPR 时读取。
- In-house RF 使用现有 `per_frequency` support manifest，60 个 `signal × scene × JSR` cell 等权 macro。
- Public RF 使用新的固定测试池 manifest：前 20 个正常 measurement record 组成 support pool，其余 record 永久作为 test；seed=111、`per_frequency` support 选 16 张图，15 个 `signal × strength` cell 等权 macro。
- OFDMA 首先按官方预处理进行子载波聚合和 letterbox；每个 target scene 的 1/2/4-shot support 独立校准，21 个 SU 的图像分数取 max 后再计算 observation-level 指标，并对 scene 等权 macro。
- 统计方法没有训练参数、没有 GPU 依赖；实验脚本采用单进程、逐图读取，输出目录边运行边保存 `metrics_per_cell.csv`、`metrics_macro.csv` 和 score archive。

入口：

```bash
python tools/eval_traditional_spectral_baselines.py \
  --protocol rf_target \
  --output-root analysis_outputs/20260805_traditional_rf_target_formal \
  --normal-sampling per_frequency \
  --support-manifest analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json

python tools/build_public_rf_support_manifest.py \
  --output analysis_outputs/20260805_public_rf_fixed_test_pool/support_manifest.json \
  --seed 111 --normal-sampling per_frequency

python tools/eval_traditional_spectral_baselines.py \
  --protocol public_rf \
  --output-root analysis_outputs/20260805_traditional_public_rf_formal \
  --normal-sampling per_frequency \
  --support-manifest analysis_outputs/20260805_public_rf_fixed_test_pool/support_manifest.json \
  --support-seed 111
```

## 3. 频谱专用 paired TTA

普通图像的翻转、任意旋转不符合频谱坐标语义，因此当前新增候选只沿物理轴做小扰动，
并且 support gallery 与 query 使用同一变换，避免把增强后的 support 当成额外异常样本。

| 数据 | 频率轴 | 时间轴 | 候选变换 |
|---|---|---|---|
| RF | x | y | 原图、时间平移 ±4 px、低功率背景噪声底抖动 |
| OFDMA | y | x | 原图、时间平移 ±4 px、低功率背景噪声底抖动 |

当前四视图候选的具体定义如下：

| 数据 | 视图 1 | 视图 2 | 视图 3 | 视图 4 |
|---|---|---|---|---|
| RF | 原图 | 时间轴上移 4 px | 时间轴下移 4 px | 背景噪声底抖动 |
| OFDMA | 原图 | 时间轴左移 4 px | 时间轴右移 4 px | 背景噪声底抖动 |

这里的背景噪声底抖动只修改每张图灰度值低于 65% 分位数的背景区域，采用确定性的平滑、
零均值局部扰动，默认强度为 3 个灰度级；高功率信号结构不直接修改。这样做是为了模拟
接收噪声底和环境背景变化，而不是把真正的信号能量整体放大。四视图中的噪声扰动同时包含
正、负局部变化，避免 max 融合偏好一张被整体抬亮的图。

四视图的原因是：时间双向平移覆盖 STFT 窗口起点或采集对齐变化；背景抖动覆盖噪声底和
接收状态变化；原图保留主要信号结构。它们都属于 TTA，因为只改变 support/query 的输入
视图，不更新模型参数、不使用异常样本，也不使用测试批次统计量。

当前实施状态如下：

| 方向 | 实施状态 | 当前证据 |
|---|---|---|
| 时间方向变化 | 已实现并完成 RF/Public RF/OFDMA smoke | 新 bundle 使用 RF 的 y 轴、OFDMA 的 x 轴 |
| 背景噪声底抖动 | 已实现并完成预览与 smoke | `rf_background_noise_jitter` / `ofdma_background_noise_jitter` |
| 频率方向变化/局部平滑 | 已实现 | 保留为历史候选，不放入当前四视图 |
| 频带宽度变化 | 移出候选 | 容易削弱对宽带异常的检测，不再纳入新方案 |
| 功率残差 | 不纳入 TTA | 它是独立的功率异常分支，不是成对测试时增强 |

因此，当前频谱专用 TTA 的新增设计是“时间双向平移 + 背景噪声底抖动”；频率平移、频率
平滑、频带宽度变化和功率残差保留为历史或独立探索，不与当前四视图混合。

代码入口是 `utils/spectral_tta.py`，ViT gallery evaluator 通过显式参数启用：

```bash
python tools/eval_cls_vit_patchcore_gallery.py \
  --paired-tta rf_spectral_time_background_v1 \
  --paired-tta-background-noise-strength 3
```

OFDMA target-scene 入口的探索命令为：

```bash
python tools/eval_cls_ofdma_fewshot_comparison.py \
  --paired-tta ofdma_spectral_time_background_v1 \
  --paired-tta-background-noise-strength 3
```

正式 target-scene 入口仍保持冻结配置；上面命令用于独立的探索性 bundle 对照。

当前保留的 bundle 包括：

- `rf_spectral_time_background_v1`：原图/时间双向平移/背景噪声底抖动；当前四视图候选；
- `ofdma_spectral_time_background_v1`：OFDMA 坐标下的原图/时间双向平移/背景噪声底抖动；
  当前四视图候选；
- `rf_spectral_structure_v1` / `ofdma_spectral_structure_v1`：频率/时间变化 + 频率平滑，
  历史结构候选；
- `rf_spectral_physics_v1/v2`：历史候选，分别含功率 gain/loss 或带宽变化，不再作为新方案。

候选只在显式参数下运行，默认的 `stft_shift_blur`/`ofdma_time_shift_blur` 不改变。当前
受控 smoke 结果显示：RF 新候选 95.41% AUROC 低于旧版 99.48%，Public RF burst 从
87.61% 小幅升至 87.65%，OFDMA `test_000` 1-shot smoke 从 100.00% 降至 95.00%。
因此它目前是有明确物理理由的探索设计，还不能替换正式主线。没有独立 validation 时，
不能把 test 上偶然最优的 bundle 直接升级为论文主线；若后续有 validation split，则用
validation macro AUROC 选 bundle，再锁定到 test。

## 4. 当前实验产物

- In-house RF 传统统计全量：`analysis_outputs/20260805_traditional_rf_target_formal/`。
- Public RF 固定测试池：`analysis_outputs/20260805_public_rf_fixed_test_pool/`。
- Public RF 传统统计全量：`analysis_outputs/20260805_traditional_public_rf_formal/`。
- Public RF `k-per-frequency` 正式少样本主实验清单与全量结果：
  [`analysis_outputs/20260810_public_rf_k_per_frequency/`](../../analysis_outputs/20260810_public_rf_k_per_frequency/)，其中
  `k=1/2/4` 分别对应 16/32/64 张 support 图像，测试正常集固定为 5120 张。
- OFDMA 传统统计全量：`analysis_outputs/20260805_traditional_ofdma_formal/`；KLD/ICA、IAD-PER、UDMA 的同协议正式结果分别见
  `analysis_outputs/20260811_kld_ica_ofdma_v2_realistic_formal/`、
  `analysis_outputs/20260811_iad_per_ofdma_v2_realistic_formal/` 和
  `analysis_outputs/20260811_udma_resnet18_ofdma_v2_realistic_formal/`。
- RF/FedJam 新增基线结果见 `analysis_outputs/20260811_kld_ica_rf_target_formal/`、
  `analysis_outputs/20260811_iad_per_rf_target_formal/`、`analysis_outputs/20260811_udma_resnet18_rf_target_formal/`、
  `analysis_outputs/20260811_kld_ica_fedjam_formal/`、`analysis_outputs/20260811_iad_per_fedjam_formal/` 和
  `analysis_outputs/20260811_udma_resnet18_fedjam_formal/`。
- FedJam 1/2/4-shot：`analysis_outputs/20260810_fedjam_traditional_fewshot_formal/`；主表采用
  `frequency_axis=1`，另有 axis-0 敏感性目录，不把两种轴向结果混合。

Public RF 的 `k-per-frequency` 传统统计结果如下。数值为 15 个
`signal type × strength` 单元的宏平均，指标按 AUROC、AUPRC、FPR@95%TPR 分列（%），
属于 PNG 频谱图上的 spectrogram-domain adaptation：

| 方法 | k=1 AUROC ↑ | k=2 AUROC ↑ | k=4 AUROC ↑ | k=1 AUPRC ↑ | k=2 AUPRC ↑ | k=4 AUPRC ↑ | k=1 FPR@95%TPR ↓ | k=2 FPR@95%TPR ↓ | k=4 FPR@95%TPR ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED [1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503) | 49.39 | 49.39 | 49.39 | 11.27 | 11.27 | 11.27 | 96.92 | 96.92 | 96.92 |
| Spectral entropy [2](https://jcs.iie.ac.cn/xxaqxb/ch/reader/view_abstract.aspx?doi=10.19363%2Fj.cnki.cn10-1380%2Ftn.2023.08.21) | 51.21 | 51.21 | 51.21 | 6.06 | 6.06 | 6.06 | 89.55 | 89.55 | 89.55 |
| Spectral flatness [3](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2016.4712) | 50.33 | 50.33 | 50.33 | 5.58 | 5.58 | 5.58 | 90.24 | 90.24 | 90.24 |
| Spectral kurtosis [4](https://www.sciencedirect.com/science/article/pii/S0888327004001517/pdf) | 53.06 | 53.06 | 53.06 | 8.11 | 8.11 | 8.11 | 92.52 | 92.52 | 92.52 |
| CA-CFAR [5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829) | 55.12 | 55.12 | 55.12 | 11.47 | 11.47 | 11.47 | 84.76 | 84.76 | 84.76 |
| SCSE（Support-Calibrated Spectral Ensemble；本项目派生，[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503)–[5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829)） | 60.05 | 61.16 | 61.34 | 9.06 | 9.23 | 9.30 | 85.05 | 82.11 | 83.58 |

完整的视觉方法同步对照表见 [`论文实验部分.md`](../paper/论文实验部分.md) 的表 1b，原始
三指标汇总见 [`full_test_metrics_summary.csv`](../../../analysis_outputs/20260810_public_rf_k_per_frequency/full_test_metrics_summary.csv)。

OFDMA 全量宏平均如下（所有方法均为 spectrogram-domain adaptation）：

| 方法 | 1-shot AUROC | 2-shot AUROC | 4-shot AUROC | 1-shot AUPRC | 2-shot AUPRC | 4-shot AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| ED [1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503) | 62.92 | 62.92 | 62.92 | 68.83 | 68.83 | 68.83 |
| Spectral entropy [2](https://jcs.iie.ac.cn/xxaqxb/ch/reader/view_abstract.aspx?doi=10.19363%2Fj.cnki.cn10-1380%2Ftn.2023.08.21) | 45.41 | 45.41 | 45.41 | 48.37 | 48.37 | 48.37 |
| Spectral flatness [3](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2016.4712) | 45.65 | 45.65 | 45.65 | 48.49 | 48.49 | 48.49 |
| Spectral kurtosis [4](https://www.sciencedirect.com/science/article/pii/S0888327004001517/pdf) | 61.97 | 61.97 | 61.97 | 69.48 | 69.48 | 69.48 |
| CA-CFAR [5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829) | 56.08 | 56.08 | 56.08 | 59.17 | 59.17 | 59.17 |
| SCSE（Support-Calibrated Spectral Ensemble；本项目派生，[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503)–[5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829)） | 69.70 | 68.76 | 68.46 | 74.77 | 72.85 | 72.41 |

论文正文只引用完成且协议一致的结果；早期 smoke 目录只用于代码检查，不作为正式数值。

## 5. FedJam 少样本补充

FedJam 的传统基线使用与视觉实验相同的 1/2/4-shot benign-only support 和完整 test
split。由于本地 FedJam 样本提供的是 224×224 spectrogram 图像而非 IQ，结果仍统一称为
**spectrogram-domain adaptation**；KPI 时序没有参与本次实验。当前按视觉评估器的轴
约定将水平轴视为频率、竖直轴视为时间（`frequency_axis=1`），该约定记录在实验协议中；
同时已完成 `frequency_axis=0` 的敏感性复核，结果保存在
`analysis_outputs/20260810_fedjam_traditional_axis0_sensitivity/`，不替换主表的轴向约定。

完整结果位于 `analysis_outputs/20260810_fedjam_traditional_fewshot_formal/`，视觉与传统
基线合并说明位于 `analysis_outputs/20260810_fedjam_fewshot_formal/README.md`。三项指标统一
放在同一张表中，按 shot 和指标分列（%）：

| 方法（出处） | 1-shot AUROC ↑ | 2-shot AUROC ↑ | 4-shot AUROC ↑ | 1-shot AUPRC ↑ | 2-shot AUPRC ↑ | 4-shot AUPRC ↑ | 1-shot FPR@95%TPR ↓ | 2-shot FPR@95%TPR ↓ | 4-shot FPR@95%TPR ↓ | 测试协议 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ED（spectrogram adaptation；[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503)） | 64.93 | 64.93 | 64.93 | 84.64 | 84.64 | 84.64 | 85.72 | 85.72 | 85.72 | full-test |
| Spectral entropy（spectrogram adaptation；[2](https://jcs.iie.ac.cn/xxaqxb/ch/reader/view_abstract.aspx?doi=10.19363%2Fj.cnki.cn10-1380%2Ftn.2023.08.21)） | 69.22 | 69.22 | 69.22 | 87.59 | 87.59 | 87.59 | 85.33 | 85.33 | 85.33 | full-test |
| Spectral flatness（spectrogram adaptation；[3](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2016.4712)） | 69.51 | 69.51 | 69.51 | 87.75 | 87.75 | 87.75 | 85.06 | 85.06 | 85.06 | full-test |
| Spectral kurtosis（spectrogram adaptation；[4](https://www.sciencedirect.com/science/article/pii/S0888327004001517/pdf)） | 75.73 | 75.73 | 75.73 | 89.36 | 89.36 | 89.36 | 71.39 | 71.39 | 71.39 | full-test |
| CA-CFAR（spectrogram adaptation；[5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829)） | 30.14 | 30.14 | 30.14 | 65.71 | 65.71 | 65.71 | 99.94 | 99.94 | 99.94 | full-test |
| SCSE（Support-Calibrated Spectral Ensemble；本项目派生，[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503)–[5](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829)） | 63.51 | 68.79 | 76.52 | 84.96 | 87.38 | 90.30 | 100.00 | 86.89 | 74.89 | full-test |

这些传统方法的输入和算法定义与需要 IQ/时间快拍的完整通信检测器不同，不能把上述
数值表述为对原始通信检测器的无条件复现。

<!-- Numeric citation links used throughout the document; each citation opens the corresponding PDF or official full-text page. -->
[1]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1447503
[2]: https://jcs.iie.ac.cn/xxaqxb/ch/reader/view_abstract.aspx?doi=10.19363%2Fj.cnki.cn10-1380%2Ftn.2023.08.21
[3]: https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2016.4712
[4]: https://www.sciencedirect.com/science/article/pii/S0888327004001517/pdf
[5]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4102829
[6]: https://spj.science.org/doi/pdf/10.34133/2022/9865016
[7]: https://arxiv.org/pdf/1807.08316
[8]: https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf
[9]: https://arxiv.org/pdf/2011.08785
[10]: https://arxiv.org/pdf/2103.04257
[11]: https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf
[12]: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
[13]: https://arxiv.org/pdf/2508.09369
