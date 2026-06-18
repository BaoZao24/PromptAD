# Score Fusion Adapter 实验方案

本文档说明 `VCPA + visual adapter + score_img 融合头` 应该如何在 **RF 跨库协议** 下验证。

重要修正：这里的跨库实验不是 `dsss_signal -> burst_signal`、`burst_signal -> chirp_signal` 这种同一项目数据内部的 pairwise transfer。根据 [RF跨库实验说明](RF跨库实验说明.md)，正确协议是：

```text
训练集：RF_SPE_PNG 公开 RF 频谱数据集派生出的 rf_signal_public_train_smoke/train
测试集：项目原始 RF 数据集 {burst, chirp, dsss} 派生出的 rf_signal/test
```

目标域 `rf_signal` 的 train split 不参与跨库训练。目标域异常样本也不参与训练。

## 1. 正确数据划分

| 阶段 | 数据集名 | 路径 | split | 是否参与参数更新 |
|---|---|---|---|---|
| 训练 | `rf_signal_public_train` | `dataset/rf_signal_public_train_smoke` | `train` | 是 |
| 训练中验证 | `rf_signal` | `dataset/rf_signal` | `test` | 否，可用 `--skip_val` 关闭 |
| 最终测试 | `rf_signal` | `dataset/rf_signal` | `test` | 否 |

源域训练集包括三类：

```text
burst / chirp / dsss
```

每类包含：

```text
300 normal + 300 abnormal
```

目标域测试集也包括：

```text
burst / chirp / dsss
```

但跨库实验只读 `test` split，不读目标域 `train` split。

## 2. 不要再使用的错误协议

下面这类命令不符合 RF 跨库协议：

```bash
python train_cross_cls.py \
  --source-dataset dsss_signal \
  --target-dataset burst_signal
```

原因：

- `dsss_signal` 不是跨库协议的源域训练集；
- `burst_signal` 不是单独的目标测试域；
- 当前 RF 跨库目标域应是完整 `rf_signal/test`，同时包含 burst / chirp / dsss；
- `train_cross_cls.py` 当前还会构造 target train gallery，这也不符合“目标域只测试、不参与训练”的 zero-shot 跨库约定。

因此，`train_cross_cls.py` 只能保留为旧的 pairwise ablation 工具，不能作为这份 RF 跨库协议的主实验入口。

## 3. 本次 adapter 代码改动的含义

本次代码已经实现了三个可训练小模块的基础能力：

- `VCPA`：正常视觉原型映射为 soft prompt token；
- `visual adapter`：对 CLIP 视觉特征做轻量残差适配；
- `score_img 融合头`：学习融合 `text score` 和 `visual score`，减少手工调 `alpha/beta/gamma`。

这些思想仍然适用于 RF 跨库协议，但实验入口必须使用正确的数据流：

```text
rf_signal_public_train_smoke/train
  -> 训练 prompt / VCPA / visual adapter / fusion head

rf_signal/test
  -> 只评估 px / sp / image-level 指标
```

## 4. 实验命令模板

以下命令来自 [RF跨库实验说明](RF跨库实验说明.md)。先用它确认基础跨库流程，再接入 score fusion / adapter 改造。

### 4.1 训练：公开 RF 源域

```bash
python train.py \
  --dataset rf_signal_public_train \
  --train_data_path dataset/rf_signal_public_train_smoke \
  --val_data_path   dataset/rf_signal \
  --save_path       results/rf_signal_cross_public_smoke_small \
  --pretrained_path /home/wangbei/.cache/clip/ViT-L-14-336px.pt \
  --prompt_len 2 --deep_prompt_len 1 \
  --features_list 6 12 18 24 \
  --pretrained openai \
  --image_size 336 --batch_size 8 \
  --epoch 1 --group_id_list 0 \
  --learning_rate 4e-5 --seed 111 \
  --device_id 2 \
  --config_path ./models/model_configs/ViT-L-14-336.json \
  --model ViT-L-14-336 \
  --skip_val
```

### 4.2 测试：目标 RF test split

```bash
python test.py \
  --dataset rf_signal \
  --data_path dataset/rf_signal \
  --checkpoint_path results/rf_signal_cross_public_smoke_small/epoch_1_group_id_0.pth \
  --save_path       results/rf_signal_cross_public_smoke_small_eval \
  --pretrained_path /home/wangbei/.cache/clip/ViT-L-14-336px.pt \
  --prompt_len 2 --deep_prompt_len 1 \
  --features_list 6 12 18 24 \
  --pretrained openai \
  --image_size 336 --seed 111 --device_id 2 \
  --config_path ./models/model_configs/ViT-L-14-336.json \
  --model ViT-L-14-336 \
  --skip_vis
```

## 5. 推荐 ablation 顺序

先固定正确数据协议，再做模块消融：

| ID | VCPA | Visual adapter | Fusion head | 目的 |
|---|---|---|---|---|
| A | 否 | 否 | 否 | 基线 |
| B | 否 | 否 | 是 | 验证 `score_img` 融合头 |
| C | 是 | 否 | 是 | 验证 VCPA |
| D | 是，多正常原型 | 否 | 是 | 验证多个正常原型 |
| E | 是，多正常原型 | 是 | 是 | 验证 visual adapter 是否通过 final score 起作用 |

每个设置都必须使用：

```text
train = rf_signal_public_train_smoke/train
test  = rf_signal/test
```

不要把 `burst/chirp/dsss` 互相当成 source/target。

## 6. 指标记录

每个实验至少记录：

- checkpoint path；
- target `rf_signal/test` 上的 pixel-level 指标；
- target `rf_signal/test` 上的 sample/image-level 指标；
- burst / chirp / dsss 三类分项结果；
- overall 平均结果。

建议结果表：

```text
analysis_outputs/score_fusion_adapter_cross_public/results.csv
```

字段：

```text
method,seed,epoch,checkpoint,px_ap,sp_ap,burst_px,chirp_px,dsss_px,burst_sp,chirp_sp,dsss_sp,notes
```

## 7. 判断标准

推进标准：

- B 相比 A 提升或至少不下降，说明 `score_img` 融合头合理；
- C/D 相比 B 提升，说明 VCPA 正常原型有价值；
- E 相比 D 提升，说明 visual adapter 在 final score 路径里被有效利用。

回滚标准：

- B 明显下降：融合头过强，降低融合头残差强度或学习率；
- D 不如 C：多个正常原型引入噪声，退回 mean 或减少 prototype 数量；
- E 不如 D：visual adapter 过强，降低 adapter alpha 或学习率。
