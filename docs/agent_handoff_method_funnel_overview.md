# Agent Handoff: PromptAD Method Funnel Overview

更新时间：2026-06-28

当前状态详见：

```text
docs/agent_handoff_method_funnel_status.md
```

项目根目录：

```text
/mnt/data/wangbei/PromptAD
```

## 1. 目标

在现有 PromptAD 框架上连续尝试小模块，直到找到稳定超过 `rf + rgb` 的方法。

所有方法必须先走同一个筛选协议：

```text
pooled universal training
burst + chirp + dsss + pulse
normal-only training
20 epoch
cls + seg
```

## 2. 新代码入口

已新增：

```text
train_rf_target_pooled_universal.py
```

这个脚本做一件事：

```text
把 48 个 cell 的 train normal 合并训练一个 universal checkpoint，
再分别评估 48 个 test cell。
```

支持方法：

```text
pooled_rf_rgb
pooled_rf_rgb_vcpa
pooled_rf_rgb_visual_adapter
pooled_rf_rgb_grouped_meanmax
```

任务：

```text
--task cls
--task seg
```

## 3. 固定协议

训练集：

```text
4 anomaly types × 4 scenes × 3 JSR = 48 cells
每个 cell 取 75% normal
全部合并成 pooled train
```

测试集：

```text
每个 cell 单独测试
test = 25% normal + 当前 cell abnormal
```

固定数据范围：

```text
burst_signal: m10db, m20db, m30db
chirp_signal: m10db, m20db, m30db
dsss_signal : m10db, m20db, m30db
pulse_signal: m20db, m30db, m40db
```

固定参数：

```text
epochs = 20
seed = 111
batch_size = 400
input_mode = rgb
normal_train_ratio = 0.75
seg eval_every = 5
```

## 4. 方法执行顺序

按下面顺序跑：

```text
1. pooled_rf_rgb                 baseline
2. pooled_rf_rgb_vcpa            VCPA
3. pooled_rf_rgb_visual_adapter  visual adapter
4. pooled_rf_rgb_grouped_meanmax grouped abnormal prototypes
```

`pooled_rf_rgb_object_agnostic` 已移出本轮 funnel：在 pooled universal 设定下 baseline 本身已经是 object-agnostic RF prompt，它没有可消融的类别专属信息。

如果前 4 条都不超过 baseline，再进入需要新增结构的线：

```text
6. anople_prompt_interaction
7. april_text_aligned_projection
8. mvfa_visual_encoder_adapter
```

## 5. 输出目录

统一输出到：

```text
analysis_outputs/20260627_method_funnel/
```

每个方法会输出：

```text
results_cls_<method>.csv
results_seg_<method>.csv
runs/<method>/cls/checkpoint/overall-best.pt
runs/<method>/seg/checkpoint/overall-best.pt
```

## 6. 汇总要求

每条方法都必须给出：

- `cls` overall 和四类平均
- `seg` overall 和四类平均
- 相对 `pooled_rf_rgb` 的 delta
- `dsss` 单独变化
- `pulse m40` 单独变化
- 是否值得进入 50 epoch 复验

## 7. 判断标准

进入下一阶段的最低条件：

```text
cls overall 或 seg overall 至少一项 >= baseline
且 dsss 不明显下降
且任一类下降不超过 1.0
```

如果 overall 只提升小于 `0.2`，视为基本持平，需要看 dsss 和 pulse m40 是否有实际收益。

## 8. 禁止事项

- 不要跑 per-dataset 单独训练。
- 不要把旧 `runs_epoch20/rf_rgb` 当 pooled baseline。
- 不要使用 public source / cross-library。
- 不要把 abnormal 放入 train。
- 不要保存多个 per-class best checkpoint 当主结果。
