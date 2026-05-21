# 实验结果索引

这个文档用于快速定位当前保留在根目录的关键结果。低优先级、临时和负结果 raw 目录已经移动到 `archive_results/`，但汇总 CSV 仍保留在 `analysis_outputs/`。

## 当前推荐结论

当前最清晰的 selection 规则是：

| signal type | selected feature |
|---|---|
| `burst_signal` | `spectral_gradient` |
| `chirp_signal` | `spectral_gradient` |
| `dsss_signal` | `dsss_weak_residual` |
| `wideband_pulse` | `rgb`，当前不采用 `spectral_gradient` |

注意：`dsss_weak_residual` 目前只完成 `seed=111`，最终论文结果还需要 `seed=222/333` 复验并报告 `mean ± std`。

## 目录说明

### `analysis_outputs/`

最重要的汇总结果都在这里。日常看结果优先看这个目录，不需要翻 raw result。

| path | 内容 | 结论 |
|---|---|---|
| `analysis_outputs/rf_four_way_compare/summary.csv` | prompt/input 四组主线对比 | `rf_signal_structured + signal_adaptive` overall 最高 `86.7625`，但只比 `rf + signal_adaptive` 高 `0.0289` |
| `analysis_outputs/rf_signal_adaptive_vs_rf_rgb/summary.csv` | `rf + signal_adaptive` vs `rf + rgb` | overall `+1.8653`，burst `+1.6683`，chirp `+3.9275`，DSSS 持平 |
| `analysis_outputs/dsss_weak_residual_all_scenes/dsss_weak_residual_vs_rgb_and_statistical.csv` | DSSS 新主结果 | `dsss_weak_residual` overall `94.5833`，比 RGB `+5.0208` |
| `analysis_outputs/wideband_pulse_feature_compare/rf_spectral_gradient_vs_rgb.csv` | wideband pulse baseline 对比 | `spectral_gradient` overall 下降 `-6.0483`，当前保留 RGB |
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

`rf + signal_adaptive` 相对 `rf + rgb/none`：

| dataset | baseline | improved | delta |
|---|---:|---:|---:|
| `burst_signal` | 86.3117 | 87.9800 | +1.6683 |
| `chirp_signal` | 78.7308 | 82.6583 | +3.9275 |
| `dsss_signal` | 89.5625 | 89.5625 | +0.0000 |
| overall | 84.8683 | 86.7336 | +1.8653 |

### DSSS weak residual

`dsss_weak_residual` 相对 RGB baseline：

| scene | RGB baseline | `dsss_weak_residual` | delta |
|---|---:|---:|---:|
| `WeaponMuseum_spectrum` | 90.5067 | 92.3700 | +1.8633 |
| `Playground_spectrum` | 89.2433 | 97.9967 | +8.7533 |
| `TimeSquare_spectrum` | 92.1600 | 97.2933 | +5.1333 |
| `Gymnasium_spectrum` | 86.3400 | 90.6733 | +4.3333 |
| ALL | 89.5625 | 94.5833 | +5.0208 |

### Wideband pulse

`rf + spectral_gradient` 相对 `rf + rgb`：

| group | RGB baseline | `spectral_gradient` | delta |
|---|---:|---:|---:|
| `m20db` mean | 99.1225 | 97.2800 | -1.8425 |
| `m30db` mean | 98.2400 | 99.3050 | +1.0650 |
| `m40db` mean | 91.0375 | 73.6700 | -17.3675 |
| overall | 96.1333 | 90.0850 | -6.0483 |

结论：`wideband_pulse` 不能直接归到 burst/chirp 的 `spectral_gradient` 路线。当前 `spectral_gradient` 在 m40db 低信噪比下明显破坏性能，因此 wideband 先保留 RGB baseline。

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
