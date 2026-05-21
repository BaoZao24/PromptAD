# 归档结果说明

这里存放低优先级、临时或负结果 raw directories。移动到这里是为了让项目根目录保持清爽；这些结果没有删除。

## `quick_checks/`

快速验证、smoke test、单点检查和临时实验。

包括：

| directory | 类型 |
|---|---|
| `result_quick_*` | 早期 quick ablation |
| `tmp_*` | smoke test 或单点功能检查 |

这些目录主要用于确认代码路径能跑通，不作为论文主结果。

## `exploratory_prompt_and_fusion/`

早期 prompt、fusion、same-scene retry 等探索性实验。

包括：

| directory | 类型 |
|---|---|
| `result_prompt_*` | prompt-only 或 prompt priority 探索 |
| `result_same_scene_prompt_retry*` | prompt retry 实验 |
| `result_fusion_ablation*` | 视觉/分数融合探索 |
| `result_split_prompt` | split 协议下 prompt 对比 raw results |

这些实验帮助排除了“只改 prompt 就能稳定提升”的路线。最终结论优先看 `analysis_outputs/prompt_three_way_compare/prompt_three_way_compare.csv` 和 `improve.md`。

## `side_or_negative_raw_results/`

有记录价值但不作为当前默认方法的 raw results。

包括：

| directory | 结论 |
|---|---|
| `result_stat_fusion` | raw spectral-stat score fusion 整体下降 |
| `result_visual_adapter` | Visual Adapter 与 baseline 基本持平 |
| `result_visual_lora` | LoRA overall 小升但不稳定 |
| `result_weaponmuseum_dsss_energy_features` | WeaponMuseum DSSS energy features 不如 `dsss_weak_residual` |

这些结果对应的汇总 CSV 仍在 `analysis_outputs/`。

## `legacy_raw_results/`

旧版或默认输出目录。

包括：

| directory | 说明 |
|---|---|
| `result` | 早期默认 raw result 目录，当前不作为主线查看入口 |

## 当前根目录保留的 raw results

根目录只保留当前还经常需要引用的 raw results：

| directory | 作用 |
|---|---|
| `result_split_ablation` | 主线 36 组/四组对比 raw results |
| `result_dsss_weak_residual_all_scenes` | DSSS weak residual 三个补充场景 raw results |
| `result_weapon_dsss_input_candidates` | WeaponMuseum DSSS candidate raw results |

日常查看优先看 `analysis_outputs/` 和 `docs/RESULTS_INDEX.md`。
