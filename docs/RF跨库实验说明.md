# RF 跨库异常分割实验：训练集与测试集说明

本实验在 VCP-CLIP 框架下做的是 **跨数据集（cross-dataset）零样本异常分割**：
源域 RF 公开数据集训练 prompt 相关参数，目标域 RF 数据集只用于评估，全程 **不用目标域参与训练**。

## 1. 源域：训练集

- **数据来源**：`/mnt/data/wangbei/data/RF_SPE_PNG/`
- **正常样本**：`/mnt/data/wangbei/data/RF_SPE_PNG/RF_Spectrum_Public_Dataset/`
  下所有 PNG 频谱 patch；不含 mask。
- **异常样本**：`/mnt/data/wangbei/data/RF_SPE_PNG/{burst, chirp, dsss}/abnormal/<snr>/*_abnormal.png`
- **异常 mask**：`/mnt/data/wangbei/data/RF_SPE_PNG/{burst, chirp, dsss}/groundtruth/<snr>/*_groundtruth.png`
  - 与 abnormal 图同名，仅后缀替换：`*_abnormal.png` → `*_groundtruth.png`
  - mask 以非零像素当作异常区域，dataloader 内部会二值化
- **wideband_pulse**：按要求忽略
- **类目划分**：源域沿用三个 RF 类别 `burst / chirp / dsss`
  - 公开正常样本通过 round-robin 均分到三类，使每类都有 normal 和 abnormal
  - 仅出现在 `meta["train"]` 中，`meta["test"]` 留空

### 元数据生成
脚本：`dataset/make_rf_signal_public_meta.py`

```bash
# 小样本 smoke（每类 300 normal + 300 abnormal），跨库实验默认使用这个版本
python dataset/make_rf_signal_public_meta.py \
  --source-root /mnt/data/wangbei/data/RF_SPE_PNG \
  --output-root dataset/rf_signal_public_train_smoke \
  --meta-name   meta_rf_signal_public_train.json \
  --max-normal-per-class   300 \
  --max-abnormal-per-class 300
```

| 版本 | 每类 normal | 每类 abnormal | 每类合计 | 三类合计 |
| --- | ---: | ---: | ---: | ---: |
| 小样本 `dataset/rf_signal_public_train_smoke` | 300 | 300 | 600 | 1800 |

> 训练只读 `meta["train"]`，所以源域样本对训练全部可见。
> 不使用全量版本，13090 张图训练单 epoch 太慢；后续要加大规模时改 `--max-normal-per-class` / `--max-abnormal-per-class` 即可。


## 2. 目标域：测试集（也用于训练阶段的 zero-shot 验证）

- **数据来源**：`/mnt/data/wangbei/anomaly_detection/VCP-CLIP/dataset/rf_signal/`
  - 由 `dataset/make_rf_signal_meta.py` 从 `/mnt/data/wangbei/data/datasets/{burst,chirp,dsss}` 转换而来
  - mask 已统一存到 `dataset/rf_signal/rf_signal_masks/`
- **元数据**：`dataset/rf_signal/meta_rf_signal.json`，含 `train` / `test` 两个 split
  - 目标域的 `train` split 不参与跨库训练，仅在“同库 smoke 实验”中使用
  - 跨库实验**只读 `test` split**

### 目标域 split 规模

| split | burst (total / normal / abnormal) | chirp | dsss |
| --- | ---: | ---: | ---: |
| train（跨库实验**未使用**） | 2045 / 1516 / 529 | 2920 / 1821 / 1099 | 2040 / 1005 / 1035 |
| test（跨库评估使用） | 691 / 510 / 181 | 986 / 612 / 374 | 696 / 342 / 354 |

跨库 `test.py` 评估总样本数：`691 + 986 + 696 = 2373` 张。

## 3. 训练 / 评估时的数据流

| 阶段 | 数据集名 | 路径 | 读取的 split | 作用 |
| --- | --- | --- | --- | --- |
| 训练 | `rf_signal_public_train` | `dataset/rf_signal_public_train_smoke` | `train` | 优化 prompt / VCP 模块 |
| 训练中 zero-shot 验证（可关） | `rf_signal` | `dataset/rf_signal` | `test` | epoch 结束时计算 AP，监控泛化 |
| 离线评估 | `rf_signal` | `dataset/rf_signal` | `test` | `test.py` 输出最终 px / sp 指标 |

为加速 smoke 实验，训练加了 `--skip_val` 开关跳过训练中的目标域验证；最终指标全部由独立的 `test.py` 给出。

`utils/dataset.py` 的 `OtherDataset` 现在会按 `mode` 选 split：
- `mode="train"` → `meta["train"]`
- `mode="test"` → `meta["test"]`

`Split_Product` 对 `rf_signal_public_train` 与 `rf_signal` 都返回同一个组：
`pre/post = ["burst", "chirp", "dsss"]`，保证源域三类共同训练。

## 4. 命令模板

```bash
# 训练（小样本 smoke）
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

# 目标域评估（跨库 zero-shot）
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

## 5. 一句话总结

- **训练用**：`/mnt/data/wangbei/data/RF_SPE_PNG`（公开 RF 频谱数据集）派生出的
  `dataset/rf_signal_public_train_smoke/meta_rf_signal_public_train.json` 的 `train` split（每类 300 + 300，共 1800 张）。
- **测试用**：`/mnt/data/wangbei/data/datasets/{burst,chirp,dsss}`（项目原始 RF 数据集）派生出的
  `dataset/rf_signal/meta_rf_signal.json` 的 `test` split。
- **核心约定**：源域只参与训练，目标域只参与评估，两个数据集严格不重叠，符合 VCP-CLIP 的 zero-shot 跨库范式。
