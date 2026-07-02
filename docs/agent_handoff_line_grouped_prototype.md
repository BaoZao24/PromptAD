# Agent Handoff: Line 4 Grouped Abnormal Prototypes

## 目标

验证把不同异常描述保留成分组 abnormal prototypes，而不是平均成单个 abnormal prototype，是否更适合多异常类型。

## 方法

```text
method = pooled_rf_rgb_grouped_meanmax
```

配置由脚本内置：

```text
text_prototype_mode = grouped_meanmax
prompt_mode = rf
input_mode = rgb
```

## 命令

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb_grouped_meanmax \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

```bash
python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb_grouped_meanmax \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

## 判断

这条线主要看 dsss。若 dsss 仍差，说明问题不只是 abnormal prototype 被平均。
