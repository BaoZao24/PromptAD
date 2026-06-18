# PromptAD RF 跨库 Adapter 实验计划

本文档只描述 **PromptAD 仓库中后续要实现的正式 RF 跨库实验**。全程按下面的数据流定义实验：公开 RF_SPE_PNG 作为 source 训练库，项目 RF 数据作为 target 测试库，不引入其他项目的训练入口或提示学习参数。

## 1. 这个“跨库”到底怎么跨

这里的跨库不是 `burst -> chirp`、`dsss -> burst` 这种同一项目内部信号类型互转。

正式 RF 跨库指的是两个数据来源不同的数据集之间迁移：

```text
源库 source library:
  /mnt/data/wangbei/data/RF_SPE_PNG
  公开 RF 频谱数据集
  用于训练

目标库 target library:
  /mnt/data/wangbei/data/datasets/{burst, chirp, dsss}
  项目原始 RF 数据集
  只用于测试
```

也就是固定做：

```text
训练：公开 RF_SPE_PNG
测试：项目原始 RF 数据集
```

这才是“跨库”。它检验的是：在公开 RF 数据上学到的 prompt / adapter，能不能泛化到项目原始 RF 数据。

具体评估对象是三个 target test：

```text
RF_SPE_PNG train -> burst test
RF_SPE_PNG train -> chirp test
RF_SPE_PNG train -> dsss test
```

这里 source 始终是 `RF_SPE_PNG`，target 始终只取项目数据的 `test` split。

## 2. 正确的数据 split

源库 meta 文件：

```text
dataset/rf_signal_public_train_smoke/meta_rf_signal_public_train.json
```

只使用：

```text
meta["train"]
```

目标库由：

```text
dataset/rf_signal/meta_rf_signal.json
```

只使用：

```text
meta["test"]
```

目标域的：

```text
meta["train"]
```

不参与训练，也不用于建 gallery。目标异常样本更不能参与训练。

完整数据流是：

```text
RF_SPE_PNG public train split
  -> 训练 PromptAD 的 prompt / VCPA / visual adapter / score_img fusion head

rf_signal test split
  -> 只做最终评估
```

## 3. 当前 PromptAD 还缺什么

当前 PromptAD 已经有这些模型模块：

- `VCPA`
- `visual adapter`
- `score_img fusion head`
- 分模块学习率

但当前 PromptAD 还没有正式 RF 跨库所需的数据入口：

1. 读取 `rf_signal_public_train_smoke/meta_rf_signal_public_train.json` 的 source loader；
2. 读取 `rf_signal/meta_rf_signal.json` 且只使用 `test` split 的 target loader；
3. 一个不读取 target train、不构造 target gallery 的正式跨库训练入口。

所以当前状态是：

```text
模型模块：已有
正式 RF 跨库数据入口：未完成
正式 RF 跨库实验：暂不能直接跑
```

## 4. 实验边界

PromptAD 的 RF 跨库实验只使用 PromptAD 自己的模型模块、数据 loader 和训练入口。其他仓库的训练命令、参数体系和实验脚本不能直接写进本文档，也不能作为 PromptAD 的可运行命令。

## 5. 需要新增的 PromptAD 跨库入口

建议新增一个明确的入口，例如：

```text
train_rf_public_to_target_cls.py
```

它必须满足：

```text
source_train = rf_signal_public_train_smoke/meta["train"]
target_test  = rf_signal/meta["test"]
```

它不能做：

```text
target_train gallery
target_train validation
target abnormal training
burst/chirp/dsss 互相作为 source-target
```

## 6. 正式实验的 ablation

等 PromptAD 的正式跨库入口实现后，再跑以下消融：

| ID | VCPA | Prototype | Visual adapter | Fusion head | 目的 |
|---|---|---|---|---|---|
| A | 否 | - | 否 | 否 | 跨库 baseline |
| B | 否 | - | 否 | 是 | 验证 `score_img` 融合头 |
| C | 是 | mean | 否 | 是 | 验证 VCPA 单均值原型 |
| D | 是 | diverse-4 | 否 | 是 | 验证多正常原型 |
| E | 是 | diverse-4 | 是 | 是 | 验证 visual adapter 是否通过 final score 起作用 |
| F | 是 | diverse-4 | 是 | 是 + 分模块 lr | 验证学习率细分 |

所有实验都必须使用同一个跨库数据流：

```text
train = RF_SPE_PNG public train
test  = project rf_signal test
```

## 7. 指标记录

正式结果至少记录：

- overall pixel-level 指标；
- overall sample/image-level 指标；
- burst / chirp / dsss 三类分项；
- checkpoint path；
- seed；
- method ID；
- 是否开启 VCPA / visual adapter / fusion head；
- 关键超参。

建议结果表：

```text
analysis_outputs/promptad_rf_cross_public_adapter/results.csv
```

字段：

```text
method,seed,checkpoint,overall_px,overall_sp,burst_px,chirp_px,dsss_px,burst_sp,chirp_sp,dsss_sp,notes
```

## 8. 判断标准

推进标准：

- B 相比 A 提升或至少不下降，说明融合头值得保留；
- C/D 相比 B 提升，说明 VCPA 正常原型有价值；
- E 相比 D 提升，说明 visual adapter 在 final score 路径中有效；
- F 相比 E 提升，说明分模块学习率有价值。

回滚标准：

- B 明显低于 A：融合头过强或训练不稳定；
- D 不如 C：多正常原型引入噪声；
- E 不如 D：visual adapter 过强或过拟合源库；
- 三类分项只提升某一类且牺牲其他类：不能作为主方法。

## 9. 下一步开发任务

下一步是补齐 PromptAD 正式 RF 跨库入口：

1. 实现 public source meta loader；
2. 实现 target test meta loader；
3. 新增正式训练入口；
4. 确认训练全程不读取目标域 train split；
5. 再运行 A-F 消融。
