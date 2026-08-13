# Baseline Source Inventory

更新时间：2026-07-25

本文件记录已经放入 `references/` 的 baseline 源码，以及后续适配 RF/Spectrum 少样本 anomaly detection 的优先级。

## 已下载源码

| 方法 | 本地路径 | 来源 | 当前用途 | 优先级 |
|---|---|---|---|---|
| PatchCore | `references/patchcore-inspection/` | `https://github.com/amazon-research/patchcore-inspection.git` | 主视觉 normal-memory baseline | P0 |
| VAE | `references/vae_ism_ano/` | `https://github.com/BaoZao24/vae_ism_ano.git` | 频谱异常检测重构类 baseline | P0 |
| Deep SVDD | `references/Deep-SVDD-PyTorch/` | `https://github.com/lukasruff/Deep-SVDD-PyTorch.git` | one-class deep baseline | P1 |
| DRAEM | `references/DRAEM/` | `https://github.com/VitjanZ/DRAEM.git` | reconstruction + discriminative baseline | P1 |
| STFPM | `references/STFPM/` | `https://github.com/gdwang08/STFPM.git` | student-teacher feature matching baseline | P1 |
| PaDiM | `references/PaDiM/` | `https://github.com/taikiinoue45/PaDiM.git` | Gaussian patch distribution baseline | P1 |
| RD4AD | `references/RD4AD/` | `https://github.com/hq-deng/RD4AD.git` | reverse-distillation baseline | P1 |
| WinCLIP | `references/WinCLIP/` | `https://github.com/caoyunkang/WinCLIP.git` | zero/few-shot CLIP reference | P2 |
| AnomalyCLIP | `references/AnomalyCLIP/` | `https://github.com/zqhang/AnomalyCLIP.git` | object-agnostic CLIP reference | P2 |
| UniVAD | `references/UniVAD/` | `https://github.com/FantasticGNU/UniVAD.git` | recent training-free few-shot reference | P2 |
| FastRecon | `references/FastRecon/` | `https://github.com/FzJun26th/FastRecon.git` | reconstruction-feature complement reference | P2 |
| APRIL-GAN | `references/VAND-APRIL-GAN/` | `https://github.com/ByChelsea/VAND-APRIL-GAN.git` | CLIP dense map reference | P2 |
| SimpleNet | `references/SimpleNet/` | `https://github.com/DonaldRR/SimpleNet.git` | simple feature anomaly baseline | P2 |
| FoundAD | `references/FoundAD/` | `https://github.com/ymxlzgy/FoundAD.git` | few-shot reference | P2 |
| SAIFE | `references/saife/` | local cloned reference | adversarial-autoencoder spectrum baseline | P1 |

## 建议先适配的 baseline

1. **PatchCore official / PatchCore-style**
   - 已经有当前项目内 `tools/eval_patchcore_cls.py` 结果。
   - 后续可补官方 `references/patchcore-inspection` 的 1/2/4-shot 完整对齐。

2. **VAE**
   - 最贴近频谱异常检测文献常用重构类方法。
   - 已完成 normal-only few-shot 适配与 self/public RF、Spectrum 评估。
   - 结果目录：`analysis_outputs/20260709_vae_baseline_sampling/`。
   - 结论：VAE 在 RF self/public 上明显弱于 PatchCore-style CNN normal-memory 和当前双视觉校准融合；可作为补充 baseline，不作为主 baseline。

3. **Deep SVDD**
   - normal-only one-class deep baseline。
   - 已完成 normal-only few-shot 适配与 self/public RF、Spectrum 评估。
   - 结果目录：`analysis_outputs/20260709_deepsvdd_baseline_sampling/`。
   - 结论：Deep SVDD 比 VAE 在部分 RF per_frequency 上更好，但整体仍明显弱于 PatchCore-style CNN normal-memory 和当前双视觉校准融合；可作为 one-class deep baseline 补充对照。

4. **DRAEM / STFPM / PaDiM / RD4AD**
   - 都是经典工业视觉 AD baseline。
   - PaDiM-style baseline 已完成主实验口径评估。
   - PaDiM 结果目录：`analysis_outputs/20260709_padim_baseline_main/`。
   - PaDiM 结论：强于 VAE/Deep SVDD，但仍低于 PatchCore-style CNN normal-memory 和当前双视觉校准融合。
   - STFPM baseline 已完成主实验口径评估。
   - STFPM 结果目录：`analysis_outputs/20260709_stfpm_baseline_main/`。
   - STFPM 结论：self RF 和 Spectrum 中等，public RF 明显不稳；可作为 teacher-student feature matching baseline 补充对照。
   - DRAEM / RD4AD 适合作为额外视觉对照，但协议适配成本高于 PatchCore/VAE/PaDiM/STFPM。

5. **WinCLIP / AnomalyCLIP / UniVAD**
   - 用于说明 prompt/CLIP 类方法在 RF 频谱图上的表现。
   - WinCLIP reference 已完成主实验口径评估。
   - WinCLIP 结果目录：`analysis_outputs/20260709_winclip_reference_main/`。
   - WinCLIP 结论：Spectrum 上较强，但 RF self/public 明显弱于当前方法；适合作为 CLIP/few-shot anomaly detection 参考项。
   - 注意很多方法是 zero-shot 或需要 auxiliary 数据，不能直接和 normal-only few-shot 主协议混用；只能作为 reference baseline。

## 当前主对比建议（四个数据集统一口径）

主实验表：

```text
ViT-only                         # internal branch ablation
CNN-only                         # internal branch ablation
Ours: Confidence Fusion          # proposed method
VAE
SAIFE
Deep SVDD
PaDiM / STFPM
WinCLIP
PatchCore-style CNN baseline
ED / spectral entropy / spectral flatness
spectral kurtosis / CA-CFAR
Ours: Confidence Fusion
```

In-house RF、Public RF、OFDMA 和 FedJam 的统一主比较集合为 ED、CA-CFAR、SCSE
(Support-Calibrated Spectral Ensemble)、KLD-Ref、IAD-PER、SAIFE、UDMA、PatchCore、
WinCLIP 和 Ours。谱熵、谱平坦度、谱峭度是 SCSE 的组件，ICA-Frozen、VAE-MSE、Deep
SVDD、PaDiM 和 STFPM 的完整结果保留为补充记录；具体表格按数据集分别列出，不把内部
消融当作外部方法参与排名。

补充实验表：

```text
WinCLIP / AnomalyCLIP / UniVAD reference
DRAEM / STFPM / PaDiM / RD4AD visual AD reference
```

ViT normal-memory 单分支属于当前方法内部消融；传统通信统计量作为独立的
spectrogram-domain adaptation 参照，不与需要 IQ 输入的完整 cyclostationary 或
eigenvalue detector 混写。

2026-07-25 的 self RF target-scene support 重跑（同一 manifest、seed=111、60
个测试单元）已统一保存在
`analysis_outputs/20260725_target_scene_self_rf_baselines_seed111/`，覆盖
官方 PatchCore、VAE、SAIFE、Deep SVDD、PaDiM、STFPM 和 WinCLIP。

<!-- The visual-baseline citation numbers below link to the corresponding PDFs. -->
[6]: https://spj.science.org/doi/pdf/10.34133/2022/9865016
[7]: https://arxiv.org/pdf/1807.08316
[8]: https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf
[9]: https://arxiv.org/pdf/2011.08785
[10]: https://arxiv.org/pdf/2103.04257
[11]: https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf
[12]: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf

## 传统通信/频谱统计出处

定量实现和输入限制见 [`docs/传统通信基线与频谱TTA.md`](传统通信基线与频谱TTA.md)。
主要出处为：Urkowitz (1967) energy detection、Rohling (1983) CA-CFAR、Antoni
(2006) spectral kurtosis、Gurugopinath (2017) spectral flatness，以及频谱熵检测
工作（2023）。当前 PNG-only 评估不把 Gardner (1991) cyclostationary detection 或
eigenvalue sensing 作为可直接复现的主表数值。

## 注意事项

- `references/` 下源码一般不提交到本仓库，由 `.gitignore` 管理。
- 每个 baseline 需要单独写 adapter，统一输出 `image AUROC`、按异常类型/JSR 的明细和 macro average。
- 不同 baseline 不允许使用异常样本参与训练或调参，除非明确标注为非主协议参考。
