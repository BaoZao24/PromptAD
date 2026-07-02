# Agent Handoff: Line 0 Pooled RF + RGB Baseline

## 目标

建立本轮 method funnel 的正式 pooled baseline。

旧结果：

```text
analysis_outputs/20260626_dataset_update_rerun/runs_epoch20/rf_rgb
analysis_outputs/20260626_dataset_update_rerun/runs_seg_epoch20_eval5/rf_rgb
```

这些是 per-dataset / per-cell 训练结果，只能参考，不能作为本轮 pooled universal baseline。

## 方法

```text
method = pooled_rf_rgb
```

配置：

```text
prompt_mode = rf
input_mode = rgb
visual_class_prompt = False
visual_adapter = False
text_prototype_mode = single
```

## 命令

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

```bash
python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

## 输出

```text
analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv
analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb.csv
analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt
analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb/seg/checkpoint/overall-best.pt
```

## 判断

所有后续方法都只和这条 pooled baseline 比较。
