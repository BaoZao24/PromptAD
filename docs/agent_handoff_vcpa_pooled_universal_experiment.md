# Agent Handoff: VCPA Pooled Universal Experiment

更新时间：2026-06-27

项目根目录：

```text
/mnt/data/wangbei/PromptAD
```

## 1. 实验目标

本轮重新验证 VCPA，但必须使用**通用 pooled 训练协议**，不要再跑 per-dataset 单独训练。

目标问题：

```text
在同一份 pooled normal 训练集上，rf + rgb + VCPA 是否优于 rf + rgb baseline？
```

需要同时跑：

```text
cls: Image-AUROC / i_roc
seg: Pixel-AUROC / p_roc
```

## 2. 必须使用的训练协议

本轮选择协议 B：**pooled 通用训练**。

训练集：

```text
burst_signal + chirp_signal + dsss_signal + pulse_signal
```

训练材料只允许使用 normal 样本。

对每个 `(dataset, scene, jsr)` cell：

```text
normal 样本按固定顺序切分
前 75% normal -> pooled train
后 25% normal -> 对应该 cell 的 test normal
abnormal -> 只进入对应该 cell 的 test abnormal
```

最终 pooled train 是所有 48 个 cell 的 train normal 合并：

```text
4 anomaly types × 4 scenes × 3 JSR = 48 cells
pooled_train = concat(all cell train normal)
```

禁止事项：

- 不允许把 abnormal 放进训练。
- 不允许每个异常类型单独训练一个 checkpoint。
- 不允许按 scene 或 JSR 单独训练 checkpoint。
- 不允许用 public/source 数据集。
- 不允许做 cross-library public -> target。

## 3. 测试集

测试仍然按 cell 拆开评估。

每个测试 cell：

```text
test normal = 该 cell 剩余 25% normal
test abnormal = 该 cell 当前 JSR 下全部 abnormal
```

需要输出：

- 每个 `(dataset, jsr, scene)` 的指标
- 每个 `(dataset, jsr)` 的 4 scene 平均
- 每个 dataset 的 12 cell 平均
- 48 cell 总平均

## 4. 数据集和 JSR 固定范围

必须包含四类：

```text
burst_signal
chirp_signal
dsss_signal
pulse_signal
```

场景固定：

```text
WeaponMuseum_spectrum
Playground_spectrum
TimeSquare_spectrum
Gymnasium_spectrum
```

JSR 固定：

```text
burst_signal: m10db, m20db, m30db
chirp_signal: m10db, m20db, m30db
dsss_signal : m10db, m20db, m30db
pulse_signal: m20db, m30db, m40db
```

## 5. 方法对照

本轮只比较两条线。

### 5.1 Baseline

方法名：

```text
pooled_rf_rgb
```

配置：

```text
prompt_mode = rf
input_mode = rgb
visual_class_prompt = False
```

### 5.2 VCPA

方法名：

```text
pooled_rf_rgb_vcpa
```

配置：

```text
prompt_mode = rf
input_mode = rgb
visual_class_prompt = True
visual_class_token_num = 2
visual_class_prompt_alpha = 0.2
visual_class_prototype_mode = mean
visual_class_prototype_num = 1
```

VCPA prototype 必须来自 pooled train normal 的 global image features：

```text
pooled train normal -> encode_image -> global_features -> mean prototype -> VCPA soft class tokens
```

不要按 dataset / scene / JSR 单独建 VCPA prototype。

## 6. 训练预算

固定参数：

```text
epochs = 20
seed = 111
batch_size = 400
split_mode = normal_75_25
normal_train_ratio = 0.75
input_mode = rgb
prompt_mode = rf
```

seg 任务固定使用：

```text
eval_every = 5
```

原因：前一轮验证中，`eval-every 5` 与 `eval-every 1` 的 p_roc 几乎一致：

```text
overall delta = -0.0071
```

记录文件：

```text
analysis_outputs/20260626_dataset_update_rerun/03_summaries/epoch20_rf_rgb_cls_seg_summary.md
```

## 7. 当前代码状态和实现要求

当前已有：

- `train_cls.py` 已有 VCPA 参数。
- `train_seg.py` 已有 VCPA 参数。
- `train_seg.py` 已支持 `--eval-every`。
- `datasets/rf_target_test_pool.py` 能按单个 signal type 汇总 4 scenes。

但本轮需要的是**四类一起 pooled 训练一个通用 checkpoint**。

因此不要直接用：

```text
run_rf_split_all.py
run_rf_split_all_seg.py
```

这两个脚本会按 cell / dataset 分开训练，不满足本轮协议。

需要先补一个 pooled universal 入口。建议命名：

```text
train_rf_target_pooled_universal.py
```

或拆成：

```text
train_rf_target_pooled_universal_cls.py
train_rf_target_pooled_universal_seg.py
```

最低实现要求：

1. 构建 pooled train dataloader：
   - 合并 burst/chirp/dsss/pulse 四类
   - 合并全部 scene
   - 合并固定 JSR
   - 只包含 75% normal train

2. 构建 48 个 test dataloader：
   - 每个 `(dataset, scene, jsr)` 一个 test loader
   - test = 25% normal + 当前 abnormal

3. 每个任务只训练一个 checkpoint：
   - `cls` 一个 pooled checkpoint
   - `seg` 一个 pooled checkpoint

4. best checkpoint 只能按 48 cell 宏平均选：
   - cls: mean `i_roc`
   - seg: mean `p_roc`

5. 不能保存 per-dataset best checkpoint 作为主结果。

## 8. 输出目录

统一输出到：

```text
analysis_outputs/20260627_vcpa_pooled_universal/
```

建议结构：

```text
analysis_outputs/20260627_vcpa_pooled_universal/
├── results_cls.csv
├── results_seg.csv
├── summary.md
├── logs/
│   ├── pooled_rf_rgb_cls.log
│   ├── pooled_rf_rgb_vcpa_cls.log
│   ├── pooled_rf_rgb_seg.log
│   └── pooled_rf_rgb_vcpa_seg.log
└── runs/
    ├── pooled_rf_rgb/
    │   ├── cls/checkpoint/overall-best.pt
    │   └── seg/checkpoint/overall-best.pt
    └── pooled_rf_rgb_vcpa/
        ├── cls/checkpoint/overall-best.pt
        └── seg/checkpoint/overall-best.pt
```

## 9. 结果表格式

`results_cls.csv` 字段：

```text
method,task,dataset,scene,jsr,i_roc,checkpoint
```

`results_seg.csv` 字段：

```text
method,task,dataset,scene,jsr,p_roc,checkpoint
```

`summary.md` 必须包含：

1. cls 每类平均和 overall
2. seg 每类平均和 overall
3. VCPA 相对 baseline 的 delta
4. dsss 单独分析
5. pulse m40 单独分析
6. 是否建议继续推进 VCPA

## 10. 判断标准

VCPA 只有在以下条件满足时，才算值得继续：

```text
cls overall >= baseline overall
seg overall >= baseline overall
dsss cls 或 seg 至少一项不能明显下降
burst/chirp/pulse 不能出现大幅回退
```

建议阈值：

```text
overall 变化小于 0.2 视为基本持平
单类下降超过 1.0 视为明显回退
```

如果 VCPA 只提升某一类，但 overall 下降，不作为主线。

## 11. 已有参考结果，不作为正式对照

下面这些结果是 per-dataset 训练，不是 pooled 通用训练，只能做参考：

```text
analysis_outputs/20260626_dataset_update_rerun/runs_epoch20/rf_rgb
analysis_outputs/20260626_dataset_update_rerun/runs_seg_epoch20/rf_rgb
analysis_outputs/20260626_dataset_update_rerun/runs_seg_epoch20_eval5/rf_rgb
```

20 epoch per-dataset 参考结果：

```text
cls rf+rgb overall = 94.0394
seg rf+rgb eval-every-5 overall = 91.7585
```

不要把这些当作本轮 pooled baseline。

## 12. 一句话执行要求

先实现 pooled universal 训练入口，然后在完全相同的 pooled train/test 协议下跑：

```text
pooled_rf_rgb
pooled_rf_rgb_vcpa
```

并分别输出 `cls` 和 `seg` 结果。只允许一个 universal checkpoint 参与每个任务的最终评估。
