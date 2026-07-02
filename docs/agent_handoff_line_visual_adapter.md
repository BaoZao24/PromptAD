# Agent Handoff: Line 2 Visual Adapter

## 目标

验证冻结 CLIP visual features 后的轻量 residual adapter 是否能改善 RF 频谱域迁移。

## 方法

```text
method = pooled_rf_rgb_visual_adapter
```

配置由 `train_rf_target_pooled_universal.py` 内置：

```text
visual_adapter = True
adapter_bottleneck_ratio = 0.25
adapter_alpha = 0.2
```

## 命令

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb_visual_adapter \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

```bash
python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb_visual_adapter \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

## 判断

这条线偏视觉域适配。若 dsss 和 pulse m40 有改善但 burst/chirp 小幅下降，可以进入 50 epoch 复验；若 overall 下降超过 1.0，停止。
