# exp_spectral_background_20260813

## Hypothesis

频率方向局部平滑在图像上变化过弱，不能很好地模拟接收环境变化。只在低功率背景区域加入小幅、平滑、可复现的噪声底扰动，可能比直接平滑频率轴更适合作为频谱 Paired-TTA。

## Change

- 新增 `rf_spectral_background_v1`：原图、频率 ±2、时间 ±4、背景噪声底升/降。
- 新增 `ofdma_spectral_background_v1`，使用 OFDMA 的时间/频率轴定义。
- 背景由每张图的低灰度分位区域近似；亮的信号区域不直接修改。
- 噪声场由固定随机种子生成，并包含空间平滑起伏与轻微颗粒成分。
- 原有 `rf_spectral_structure_v1`、`ofdma_spectral_structure_v1` 和正式默认设置保持不变。

## Smoke test

```bash
python -m py_compile \
  utils/spectral_tta.py \
  tools/preview_spectral_background_tta.py \
  tools/eval_cls_vit_patchcore_gallery.py \
  tools/eval_cls_public_rf_vit_patchcore_gallery.py \
  tools/eval_cls_ofdma_fewshot_comparison.py
python tools/preview_spectral_background_tta.py
```

## Observable result

预览覆盖 `burst_signal`、`chirp_signal`、`dsss_signal`、`pulse_signal` 和
`wideband_pulse` 的正常/异常图像，产物位于：

`analysis_outputs/20260813_spectral_background_tta_preview/`

背景噪声底升/降在图像上可见，信号主结构仍保留；这只是可视化 smoke test，尚未证明 AUROC、AUPRC 或 FPR@95%TPR 改善。

## Conclusion

该方向值得进入小规模指标实验，但在正式实验前需要重点检查：低灰度异常是否被误当成背景、不同噪声强度是否稳定，以及 RF/Public RF/OFDMA 是否同时受益。
