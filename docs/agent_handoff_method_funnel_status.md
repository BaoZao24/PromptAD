# Agent Handoff: Method Funnel Current Status

更新时间：2026-06-28

## 1. 当前结论边界

当前不要继续解读旧 `object_agnostic` 和旧 `grouped_meanmax` 结果。

原因：

```text
旧 pooled 入口使用 dataset="rf_target_test_pool", class_name="signal"
当时 is_rf_prompt_class(...) = False
导致 RF grouped/object prompt 逻辑静默回退
```

后续已修复 pooled RF 判定，并把 grouped abnormal prototypes 改为：

```text
burst / chirp / dsss / pulse
```

注意：`wideband` 不属于当前实验方案，不要再引入。

## 2. 结果状态

| 方法 | cls | seg | 状态 | 处理 |
|---|---:|---:|---|---|
| `pooled_rf_rgb` | 已完成 | 已完成 | 有效 baseline | 保留 |
| `pooled_rf_rgb_visual_adapter` | 运行中/未完整落盘 | 已完成 | 只作参考 | 等 cls，但不优先推进 |
| `pooled_rf_rgb_vcpa` | 未见结果 | 未见结果 | 待跑 | 优先补 |
| `pooled_rf_rgb_grouped_meanmax` | 旧结果无效 | 旧结果无效 | 修复后待重跑 | 优先补 |
| `pooled_rf_rgb_object_agnostic` | 旧结果无效 | 旧结果无效 | pooled 设定下无意义 | 移出 funnel |

## 3. 等当前训练结束后的第一步

先生成统一汇总表：

```bash
python tools/summarize_method_funnel.py \
  --root analysis_outputs/20260627_method_funnel
```

输出：

```text
analysis_outputs/20260627_method_funnel/summaries/method_funnel_summary.csv
analysis_outputs/20260627_method_funnel/summaries/method_funnel_delta_vs_baseline.csv
analysis_outputs/20260627_method_funnel/summaries/method_funnel_summary.md
```

## 4. 下一轮优先实验

只补两条：

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb_vcpa \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id <GPU>

python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb_vcpa \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id <GPU>

python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb_grouped_meanmax \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id <GPU>

python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb_grouped_meanmax \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id <GPU>
```

跑完后再次执行汇总脚本。

## 5. 判读标准

对每条方法只和 `pooled_rf_rgb` 比：

```text
primary: cls overall delta
secondary: seg overall delta
guardrail: burst/chirp 不明显下降
guardrail: dsss/pulse 不继续恶化
```

建议判定：

```text
overall 提升 >= 0.2: 有继续价值
overall 在 [-0.2, +0.2]: 基本持平，看 dsss/pulse 是否改善
overall 下降 < -0.2: 不推进
任一类下降超过 1.0: 谨慎，不作为主线
```

## 6. 如果 VCPA / grouped 都不提升

不要继续堆小 adapter。下一步转向分数层小 trick：

```text
1. top-k image score aggregation
2. grouped_meanmax + top-k
3. normal gallery nearest-neighbor calibration
```
