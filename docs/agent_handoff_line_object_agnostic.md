# Agent Handoff: Line 3 Object-Agnostic Prompt

## 目标

验证去掉具体类别词后，PromptAD 是否更适合 RF 频谱图。

## 方法

```text
method = pooled_rf_rgb_object_agnostic
```

配置由脚本内置：

```text
prompt_mode = rf_object_agnostic
input_mode = rgb
```

## 命令

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb_object_agnostic \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

```bash
python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb_object_agnostic \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400 \
  --gpu-id 0
```

## 判断

如果 RF 类名语义拖累模型，这条线应该改善 dsss 或 pulse m40。若四类整体持平但 dsss 明显提升，可以继续。
