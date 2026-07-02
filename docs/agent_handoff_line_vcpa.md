# Agent Handoff: Line 1 VCPA

## 目标

验证正常视觉原型注入 prompt 是否能超过 pooled `rf + rgb`。

## 方法

```text
method = pooled_rf_rgb_vcpa
```

配置由 `train_rf_target_pooled_universal.py` 内置：

```text
visual_class_prompt = True
visual_class_token_num = 2
visual_class_prompt_alpha = 0.2
visual_class_prototype_mode = mean
visual_class_prototype_num = 1
```

VCPA prototype 必须来自 pooled train normal 的 global image features。

## 命令

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb_vcpa \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

```bash
python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb_vcpa \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

## 对照

必须先跑：

```text
pooled_rf_rgb
```

只和 pooled baseline 比，不和旧 per-dataset 结果比。

## 重点观察

- dsss 是否提升
- cls overall 是否超过 baseline
- seg overall 是否超过 baseline
- burst/chirp/pulse 是否出现明显回退
