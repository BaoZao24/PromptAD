# 实验结果索引

这个文档用于快速定位当前保留在根目录的关键结果。低优先级、临时和负结果 raw 目录已经移动到 `archive_results/`，但汇总 CSV 仍保留在 `analysis_outputs/`。

Selection 方法的报告正文见：

```text
docs/SELECTION_REPORT.md
```

## 当前推荐结论

当前最清晰的 selection 规则是：

| signal type | selected feature |
|---|---|
| `burst_signal` | `optimized spectral gradient` |
| `chirp_signal` | `optimized spectral gradient` |
| `dsss_signal` | `dsss_weak_residual` |
| `wideband_pulse` | `rgb`，当前不采用 `spectral_gradient` |

`dsss_weak_residual` 已补齐 `seed=111/222/333`。三 seed 下 DSSS 12 点均值为 `95.4031 ± 0.0424`，相对 `rf + rgb` 的 `89.4981 ± 0.1087` 稳定提升。

## 目录说明

### `analysis_outputs/`

最重要的汇总结果都在这里。日常看结果优先看这个目录，不需要翻 raw result。

| path | 内容 | 结论 |
|---|---|---|
| `analysis_outputs/rf_four_way_compare/summary.csv` | prompt/input 四组主线对比 | `rf_signal_structured + signal_adaptive` overall 最高 `86.7625`，但只比 `rf + signal_adaptive` 高 `0.0289` |
| `analysis_outputs/rf_signal_adaptive_vs_rf_rgb/summary.csv` | `rf + signal_adaptive` vs `rf + rgb` | overall `+1.8653`，burst `+1.6683`，chirp `+3.9275`，DSSS 持平 |
| `analysis_outputs/multi_seed_stability/final_selection_mean_std.csv` | 三 seed 最终 selection 汇总 | `signal_adaptive_v2_weak_dsss` overall `88.6655 ± 0.1797`，比 RGB 高 `+3.8802` |
| `analysis_outputs/multi_seed_stability/dsss_inputs_mean_std.csv` | DSSS 三种输入三 seed 对比 | `dsss_weak_residual` 为 `95.4031 ± 0.0424`，稳定高于 RGB/statistical |
| `analysis_outputs/dsss_weak_residual_all_scenes/dsss_weak_residual_vs_rgb_and_statistical.csv` | DSSS 新主结果 | `dsss_weak_residual` overall `94.5833`，比 RGB `+5.0208` |
| `analysis_outputs/wideband_pulse_feature_compare/rf_spectral_gradient_vs_rgb.csv` | wideband pulse baseline 对比 | `spectral_gradient` overall 下降 `-6.0483`，当前保留 RGB |
| `analysis_outputs/wideband_pulse_feature_compare/wideband_rgb_vs_spectral_gradient_heatmaps.png` | wideband pulse 对比 heatmap | 同时展示 RGB、`spectral_gradient` 和 delta |
| `analysis_outputs/wideband_pulse_png_rgb/rf_rgb_results.csv` | `RF_SPE_PNG/wideband_pulse` RGB 结果 | m20/m30/m40 平均 `96.9467` |
| `analysis_outputs/wideband_pulse_png_rgb/rf_rgb_heatmap.png` | `RF_SPE_PNG/wideband_pulse` RGB heatmap | 单数据源三档位结果可视化 |
| `analysis_outputs/optimized_spectral_gradient_selftest_compare.csv` | 自测 burst/chirp 的优化梯度对比表 | 自测 burst `+3.1083`，chirp `+6.5675`，因此自测默认切到优化后的 spectral gradient |
| `analysis_outputs/all_anomaly_summary_tables/selftest_all_anomalies_baseline_rgb.csv` | 自测数据集所有异常 baseline 总表 | burst/chirp/dsss/wideband，RGB baseline |
| `analysis_outputs/all_anomaly_summary_tables/selftest_all_anomalies_improved_selected.csv` | 自测数据集所有异常改进后总表 | burst/chirp 用 `optimized spectral gradient`，DSSS 用 `dsss_weak_residual`，wideband 用 RGB |
| `analysis_outputs/all_anomaly_summary_tables/rf_spe_png_all_anomalies_baseline_rgb.csv` | RF_SPE_PNG/公开数据所有异常 baseline 总表 | 已汇总 burst/chirp/dsss/wideband pulse，作为当前公开数据主总表 |
| `analysis_outputs/all_anomaly_summary_tables/rf_spe_png_all_anomalies_improved_selected.csv` | RF_SPE_PNG/公开数据所有异常改进后总表 | 汇总当前形态感知 selection 版本，便于和 baseline 直接对比 |
| `analysis_outputs/rf_spe_png_new_levels/new_m40_m50_baseline_vs_selected.csv` | RF_SPE_PNG 低功率补充实验 | 保留 `m40db/m50db` 原始补充对比，其中 `m50db` 不纳入当前主表均值 |
| `analysis_outputs/dsss_statistical_vs_baseline/dsss_statistical_vs_baseline.csv` | DSSS statistical 旧候选 | overall `+1.3150`，但 WeaponMuseum 下降 |
| `analysis_outputs/weapon_dsss_input_candidates/weapon_dsss_input_candidates_vs_rgb.csv` | WeaponMuseum DSSS 多候选 | `dsss_weak_residual` 最好，CLAHE 明显失败 |
| `analysis_outputs/visual_adapter_compare/chirp_rf_signal_adaptive_adapter_vs_baseline.csv` | Visual Adapter | 基本持平，未超过 baseline |
| `analysis_outputs/visual_lora_compare/chirp_rf_signal_adaptive_lora_r4_vs_baseline.csv` | Visual LoRA | overall 小升，但不稳定，不作为默认 |
| `analysis_outputs/stat_fusion_compare/dsss_rf_signal_adaptive_stat_beta05_vs_baseline.csv` | raw stat score fusion | overall 下降，不采用 |
| `analysis_outputs/prompt_three_way_compare/prompt_three_way_compare.csv` | prompt-only 对比 | prompt 文案单独没有稳定收益 |

### `result_split_ablation/`

主线 raw results，体积最大。保留在根目录，因为它支撑核心 36 组和四组对比。

重要子目录：

| subdir | 含义 |
|---|---|
| `legacy_rgb` | 最初 baseline |
| `rf_rgb` | RF prompt + RGB/默认输入 |
| `rf_grad` | RF prompt + `spectral_gradient` |
| `rf_signal_adaptive` | RF prompt + signal-adaptive input |
| `rf_signal_structured_rgb` | structured prompt + RGB/默认输入 |
| `rf_signal_structured_signal_adaptive` | structured prompt + signal-adaptive input |
| `rf_object_agnostic_signal_adaptive` | object-agnostic prompt + signal-adaptive input |

对应汇总优先看：

```text
analysis_outputs/rf_four_way_compare/summary.csv
analysis_outputs/rf_signal_adaptive_vs_rf_rgb/summary.csv
```

### `result_dsss_weak_residual_all_scenes/`

DSSS `dsss_weak_residual` 在 `Playground_spectrum`、`TimeSquare_spectrum`、`Gymnasium_spectrum` 三个场景上的 raw results。

说明：
- `WeaponMuseum_spectrum` 的 `dsss_weak_residual` raw results 在 `result_weapon_dsss_input_candidates/`；
- 四场景合并汇总在 `analysis_outputs/dsss_weak_residual_all_scenes/dsss_weak_residual_vs_rgb_and_statistical.csv`。

### `result_weapon_dsss_input_candidates/`

WeaponMuseum DSSS 候选输入的 raw results：

| input | 结论 |
|---|---|
| `dsss_rgb_residual` | 平均略升，但 m30db 下降，不稳定 |
| `dsss_weak_residual` | 三个 ISR 都提升，保留 |
| `dsss_clahe` | 明显下降，不采用 |

对应汇总：

```text
analysis_outputs/weapon_dsss_input_candidates/weapon_dsss_input_candidates_vs_rgb.csv
```

## 主线数值

### Signal-adaptive 输入

`rf + signal_adaptive` 相对 `rf + rgb/none`。下表是 `seed=111` 单 seed 结果：

| dataset | baseline | improved | delta |
|---|---:|---:|---:|
| `burst_signal` | 86.3117 | 87.9800 | +1.6683 |
| `chirp_signal` | 78.7308 | 82.6583 | +3.9275 |
| `dsss_signal` | 89.5625 | 89.5625 | +0.0000 |
| overall | 84.8683 | 86.7336 | +1.8653 |

### DSSS weak residual

`dsss_weak_residual` 相对 RGB baseline。下表是 `seed=111` 单 seed 结果：

| scene | RGB baseline | `dsss_weak_residual` | delta |
|---|---:|---:|---:|
| `WeaponMuseum_spectrum` | 90.5067 | 92.3700 | +1.8633 |
| `Playground_spectrum` | 89.2433 | 97.9967 | +8.7533 |
| `TimeSquare_spectrum` | 92.1600 | 97.2933 | +5.1333 |
| `Gymnasium_spectrum` | 86.3400 | 90.6733 | +4.3333 |
| ALL | 89.5625 | 94.5833 | +5.0208 |

三 seed 汇总：

| method | mean Image-AUROC | std |
|---|---:|---:|
| `rf + rgb` | 89.4981 | 0.1087 |
| `rf + dsss_statistical` | 91.2033 | 0.7567 |
| `rf + dsss_weak_residual` | 95.4031 | 0.0424 |

最终 36 点 selection 汇总：

| method | mean Image-AUROC | std |
|---|---:|---:|
| `rf + rgb` | 84.7853 | 0.0750 |
| `signal_adaptive_v1` | 86.6971 | 0.1386 |
| `morphology_aware_selection` | 91.3587 | single-seed summary |

`signal_adaptive_v2` 已加入代码，可直接复跑：

```bash
python run_rf_split_all.py --gpus 0 1 2 3 --epochs 50 \
  --prompt-mode rf --input-mode signal_adaptive_v2 \
  --root-dir ./result_split_ablation/rf_signal_adaptive_v2
```

### Wideband pulse

`rf + spectral_gradient` 相对 `rf + rgb`：

| group | RGB baseline | `spectral_gradient` | delta |
|---|---:|---:|---:|
| `m20db` mean | 99.1225 | 97.2800 | -1.8425 |
| `m30db` mean | 98.2400 | 99.3050 | +1.0650 |
| `m40db` mean | 91.0375 | 73.6700 | -17.3675 |
| overall | 96.1333 | 90.0850 | -6.0483 |

结论：`wideband_pulse` 不能直接归到 burst/chirp 的 `spectral_gradient` 路线。当前 `spectral_gradient` 在 m40db 低信噪比下明显破坏性能，因此 wideband 先保留 RGB baseline。

`RF_SPE_PNG/wideband_pulse` 单数据源版本的 `rf + rgb` 结果：

| source | m20db | m30db | m40db | mean |
|---|---:|---:|---:|---:|
| `RF_SPE_PNG/wideband_pulse` | 98.1700 | 98.6300 | 94.0400 | 96.9467 |

对应 heatmap：

```text
analysis_outputs/wideband_pulse_png_rgb/rf_rgb_heatmap.png
```

对应 heatmap：

```text
analysis_outputs/wideband_pulse_feature_compare/rgb_heatmap.png
analysis_outputs/wideband_pulse_feature_compare/spectral_gradient_heatmap.png
analysis_outputs/wideband_pulse_feature_compare/delta_heatmap.png
analysis_outputs/wideband_pulse_feature_compare/wideband_rgb_vs_spectral_gradient_heatmaps.png
```

## 已归档结果

低优先级 raw results 已移动到 `archive_results/`。这些目录不是删除，只是不再放根目录干扰查看。

归档说明见：

```text
archive_results/README.md
```

## 推荐查看顺序

1. 先看 `docs/RESULTS_INDEX.md`，确认每个结果在哪里。
2. 再看 `improve.md`，了解实验背景、失败尝试和方法选择依据。
3. 需要具体数值时看 `analysis_outputs/*/*.csv`。
4. 只有需要复查单个实验输出时，才进入 `result_split_ablation/` 或 DSSS raw result 目录。
