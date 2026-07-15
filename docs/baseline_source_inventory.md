# Baseline Source Inventory

更新时间：2026-07-09

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

## 当前主对比建议

主实验表：

```text
PatchCore-style baseline
VAE
Deep SVDD
Ours: Calibrated Dual Visual Normality Fusion
```

补充实验表：

```text
PromptAD text+ViT reference
WinCLIP / AnomalyCLIP / UniVAD reference
DRAEM / STFPM / PaDiM / RD4AD visual AD reference
```

## 注意事项

- `references/` 下源码一般不提交到本仓库，由 `.gitignore` 管理。
- 每个 baseline 需要单独写 adapter，统一输出 `image AUROC`、按异常类型/JSR 的明细和 macro average。
- 不同 baseline 不允许使用异常样本参与训练或调参，除非明确标注为非主协议参考。
