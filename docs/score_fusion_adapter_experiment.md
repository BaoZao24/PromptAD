# Score Fusion Adapter 实验方案

本文档说明本次架构改造做了什么，以及如何验证 `VCPA + visual adapter + score_img 融合头` 是否真的提升跨库异常检测。

## 1. 本次代码改动

### 1.1 三个小网络同时参与训练

当前建议实验使用三类可训练模块：

- `prompt learner`：原有可学习 prompt。
- `VCPA`：把正常视觉原型映射成 soft prompt token。
- `visual adapter`：对 CLIP 视觉特征做轻量残差调整。
- `score_img 融合头`：学习如何融合 `text score` 和 `visual score`，替代手工设置 `alpha/beta/gamma`。

CLIP 主干仍然冻结，不参与训练。

### 1.2 `learnable_score_fusion` 现在是真正的 final-score supervision

旧逻辑中，融合头训练时使用的 `textual_anomaly` 是从缓存/评估函数得到的 detached 分数，主要训练融合头本身。

现在跨库训练中：

```text
text logits -> text score
visual anomaly map -> visual topk / max / gap
text score + visual features -> score_img fusion head -> score_img loss
```

`score_img loss` 可以反向影响：

- prompt learner
- VCPA
- visual adapter
- score fusion head

也就是说，训练目标和评估目标更一致。

### 1.3 VCPA 支持多个正常原型

新增参数：

```bash
--visual-class-prototype-mode mean|diverse
--visual-class-prototype-num 1|2|4|8
```

- `mean`：原始方案，所有正常样本求一个均值原型。
- `diverse`：从正常样本视觉特征中选多个彼此差异较大的原型，再通过 VCPA 生成 prompt token 后取平均。

第一轮建议测试：

```bash
--visual-class-prototype-mode diverse
--visual-class-prototype-num 4
```

### 1.4 支持不同模块学习率

新增参数：

```bash
--prompt-lr
--visual-class-prompt-lr
--visual-adapter-lr
--score-fusion-lr
--cnn-mamba-lr
--visual-lora-lr
```

如果不设置，默认都使用 `--lr`。

第一轮建议先统一学习率，验证架构方向；第二轮再细分学习率。

## 2. 推荐实验设置

基础设置沿用当前跨库主设定：

```text
prompt_mode = rf
input_mode = morph_fusion_gray_residual_a01
split_mode = normal_75_25
k-shot = 1
noise_level = m10db
seed = 111
Epoch = 50
```

## 3. 第一轮：确认融合头是否有效

目标：验证 `score_img` 监督是否比纯 text-only 更好。

### 3.1 Baseline

```bash
python train_cross_cls.py \
  --source-dataset dsss_signal \
  --target-dataset burst_signal \
  --source-class-name Gymnasium_spectrum \
  --target-class-name Gymnasium_spectrum \
  --source-noise-level m10db \
  --target-noise-level m10db \
  --split-mode normal_75_25 \
  --k-shot 1 \
  --prompt-mode rf \
  --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only \
  --Epoch 50 \
  --seed 111
```

### 3.2 只开融合头

```bash
python train_cross_cls.py \
  --source-dataset dsss_signal \
  --target-dataset burst_signal \
  --source-class-name Gymnasium_spectrum \
  --target-class-name Gymnasium_spectrum \
  --source-noise-level m10db \
  --target-noise-level m10db \
  --split-mode normal_75_25 \
  --k-shot 1 \
  --prompt-mode rf \
  --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only \
  --learnable-score-fusion True \
  --learnable-score-fusion-alpha 0.25 \
  --learnable-score-fusion-lambda 0.5 \
  --Epoch 50 \
  --seed 111
```

观察点：

- 如果 AUROC 提升，说明 `score_img loss` 方向有效。
- 如果提升很小但不下降，说明可以继续加入 VCPA/visual adapter。
- 如果下降，先降低 `--learnable-score-fusion-alpha` 到 `0.1`。

## 4. 第二轮：加入 VCPA

目标：验证正常视觉原型对 prompt 是否有帮助。

### 4.1 VCPA 单均值原型

```bash
python train_cross_cls.py \
  --source-dataset dsss_signal \
  --target-dataset burst_signal \
  --source-class-name Gymnasium_spectrum \
  --target-class-name Gymnasium_spectrum \
  --source-noise-level m10db \
  --target-noise-level m10db \
  --split-mode normal_75_25 \
  --k-shot 1 \
  --prompt-mode rf \
  --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only \
  --visual-class-prompt True \
  --visual-class-token-num 4 \
  --visual-class-prototype-mode mean \
  --learnable-score-fusion True \
  --learnable-score-fusion-alpha 0.25 \
  --learnable-score-fusion-lambda 0.5 \
  --Epoch 50 \
  --seed 111
```

### 4.2 VCPA 多正常原型

```bash
python train_cross_cls.py \
  --source-dataset dsss_signal \
  --target-dataset burst_signal \
  --source-class-name Gymnasium_spectrum \
  --target-class-name Gymnasium_spectrum \
  --source-noise-level m10db \
  --target-noise-level m10db \
  --split-mode normal_75_25 \
  --k-shot 1 \
  --prompt-mode rf \
  --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only \
  --visual-class-prompt True \
  --visual-class-token-num 4 \
  --visual-class-prototype-mode diverse \
  --visual-class-prototype-num 4 \
  --learnable-score-fusion True \
  --learnable-score-fusion-alpha 0.25 \
  --learnable-score-fusion-lambda 0.5 \
  --Epoch 50 \
  --seed 111
```

观察点：

- `diverse` 是否稳定优于 `mean`。
- 如果 `diverse=4` 不稳定，再试 `--visual-class-prototype-num 2`。
- 如果 VCPA 干扰过强，降低 `--visual-class-prompt-alpha 0.1`。

## 5. 第三轮：加入 visual adapter

目标：验证 visual adapter 在进入 `score_img` 融合头后是否开始有效。

```bash
python train_cross_cls.py \
  --source-dataset dsss_signal \
  --target-dataset burst_signal \
  --source-class-name Gymnasium_spectrum \
  --target-class-name Gymnasium_spectrum \
  --source-noise-level m10db \
  --target-noise-level m10db \
  --split-mode normal_75_25 \
  --k-shot 1 \
  --prompt-mode rf \
  --input-mode morph_fusion_gray_residual_a01 \
  --cls-score-mode text_only \
  --visual-class-prompt True \
  --visual-class-token-num 4 \
  --visual-class-prototype-mode diverse \
  --visual-class-prototype-num 4 \
  --visual-adapter True \
  --adapter-alpha 0.2 \
  --learnable-score-fusion True \
  --learnable-score-fusion-alpha 0.25 \
  --learnable-score-fusion-lambda 0.5 \
  --Epoch 50 \
  --seed 111
```

观察点：

- 如果 visual adapter 开启后提升，说明之前的问题主要是评分路径没看见它。
- 如果下降，优先试：

```bash
--adapter-alpha 0.1
--learnable-score-fusion-alpha 0.1
```

## 6. 第四轮：模块学习率细分

如果第三轮方向有效，再细分学习率：

```bash
--lr 0.002 \
--prompt-lr 0.001 \
--visual-class-prompt-lr 0.002 \
--visual-adapter-lr 0.001 \
--score-fusion-lr 0.005
```

直觉：

- prompt 更敏感，学习率小一点。
- VCPA 中等。
- visual adapter 中等偏小。
- fusion head 很小，可以稍大学快一点。

## 7. 推荐 ablation 表

先在 3 个困难 pair 上跑：

| ID | VCPA | Prototype | Visual adapter | Fusion head | 目的 |
|---|---|---|---|---|---|
| A | 否 | - | 否 | 否 | baseline |
| B | 否 | - | 否 | 是 | 验证 score_img loss |
| C | 是 | mean-4token | 否 | 是 | 验证 VCPA |
| D | 是 | diverse-4proto-4token | 否 | 是 | 验证多正常原型 |
| E | 是 | diverse-4proto-4token | 是 | 是 | 验证 visual adapter |
| F | 是 | diverse-4proto-4token | 是 | 是 + 分模块 lr | 验证学习率细分 |

建议 pair：

| Source -> Target | Class |
|---|---|
| `burst_signal -> chirp_signal` | `Gymnasium_spectrum` |
| `dsss_signal -> burst_signal` | `WeaponMuseum_spectrum` |
| `dsss_signal -> burst_signal` | `Gymnasium_spectrum` |

确认有效后再跑完整 12-case matrix。

## 8. 结果记录

每个实验记录：

- best target Image-AUROC
- best epoch
- source loss 曲线
- 每轮 target AUROC
- 对应 checkpoint 路径
- `scores/*.npz` 中的 image scores

建议新建表格：

```text
analysis_outputs/score_fusion_adapter_ablation/results.csv
```

字段：

```text
source_dataset,target_dataset,class,seed,method,best_epoch,best_i_roc,notes
```

## 9. 判断标准

继续推进的标准：

- B 相比 A 不下降，最好有稳定提升。
- D 相比 C 有提升，说明多正常原型有效。
- E 相比 D 有提升，说明 visual adapter 在新评分路径下有效。

停止或回滚的标准：

- B 明显低于 A：融合头过强或 score loss 不稳定，先降 `learnable-score-fusion-alpha`。
- E 明显低于 D：visual adapter 过强，先降 `adapter-alpha` 或单独调低 `visual-adapter-lr`。
- full matrix 只有个别 case 提升：保留为 ablation，不作为主方法。
