# Agent Handoff: RF Spectrogram Anomaly Detection Task Brief

更新时间：2026-06-05
项目根目录：`/mnt/data/wangbei/PromptAD`

## 1. 任务目标

这个文档提供给另一个完全不了解当前项目的 agent。该 agent 不需要继承当前方法，也不需要使用现有模型结构。它的任务是：

**针对射频频谱图异常检测任务，使用用户后续指定的其他异常检测方法（例如 VAE、扩散模型或其他方法）完成实验，并在固定协议下给出结果对比。**

这个文档只负责说明：

1. 要检测的任务是什么；
2. 数据集路径在哪里；
3. 训练集和测试集怎么划分；
4. 当前已经确定的实验协议是什么；
5. 需要输出哪些表格和结论；
6. 哪些实验设置已经确定，不能随意改口径。

**不要在这个任务里重新定义 baseline，不要擅自改数据协议，不要覆盖现有主表。**

---

## 2. 任务定义

任务类型：**频谱图异常检测（spectrogram anomaly detection）**

输入：射频频谱 PNG 图像

输出：图像级异常检测结果（Image-AUROC 为主）

基本约束：
- 训练阶段默认只使用 normal 样本；
- abnormal 样本仅用于测试；
- 新 agent 将由用户指定具体方法，例如 VAE、扩散模型或其他异常检测框架；
- 新方法必须在与当前固定协议一致的条件下做实验。

---

## 3. 数据集与实验协议

## 3.1 自测数据集

### 路径

```text
/mnt/data/wangbei/data/datasets/burst
/mnt/data/wangbei/data/datasets/chirp
/mnt/data/wangbei/data/datasets/dsss
```

### 场景

```text
WeaponMuseum_spectrum
Playground_spectrum
TimeSquare_spectrum
Gymnasium_spectrum
```

### 协议

```text
split_mode = normal_75_25
75% normal 样本用于训练
25% normal 样本用于测试
全部 abnormal 样本仅用于测试
seed = 111
epochs = 50
```

### 说明

- 当前自测主实验统一按 `normal_75_25` 解释；
- 不要把历史日志中的 `k_shot=1` 当成自测主实验口径；
- 如果新方法不需要 epoch 概念，也必须保持训练/测试划分与现有协议一致。

---

## 3.2 公开 RF 数据集（RF_SPE_PNG）

### 正常样本来源

```text
/mnt/data/wangbei/data/RF_SPE_PNG/RF_Spectrum_Public_Dataset/
```

### 协议

```text
k_shot = 1
split_mode = normal_75_25
```

在这个公开数据集里，协议的实际含义是：

- 按目录排序后的前 `k_shot=1` 个 `MeasRes_*` 目录做 normal train；
- 剩余 `MeasRes_*` 目录中的全部 patch 做 normal test；
- abnormal 来自各异常类型对应目录。

### 当前固定采用的异常档位协议

```text
burst abnormal = /mnt/data/wangbei/data/RF_SPE_PNG/burst/abnormal/{m30db,m40db,m50db}
chirp abnormal = /mnt/data/wangbei/data/RF_SPE_PNG/chirp/abnormal/{m50db,m55db,m60db}
dsss abnormal = /mnt/data/wangbei/data/RF_SPE_PNG/dsss/abnormal/{m20db,m30db,m40db}
seed = 111
epochs = 50
```

### 为什么这样选

- `burst` 的 `m20db` 太容易，因此改成 `m30/m40/m50`；
- `chirp` 的低档位太接近饱和，因此改成 `m50/m55/m60`；
- `dsss` 目前仍保留 `m20/m30/m40`。

### 额外说明

当前代码侧已经支持以下 noise level 字符串：

```text
m10db, m20db, m30db, m40db, m50db, m55db, m60db, m70db
```

如果新 agent 复用现有脚本读取公开数据，不需要再为 `m55db` 单独改参数层。

---

## 3.3 spectrum 数据集

### 完整路径

```text
/mnt/data/wangbei/PromptAD/datasets/spectrum
```

### 类别

```text
16QAM
CHIRP
GMSK
QPSK
```

### 训练集和测试集

对每个类别，当前固定口径是：

```text
训练集：<class>/train/good
测试正常：<class>/test/good
测试异常：<class>/test/bad
```

例如 `16QAM` 的训练/测试集是：

```text
/mnt/data/wangbei/PromptAD/datasets/spectrum/16QAM/train/good
/mnt/data/wangbei/PromptAD/datasets/spectrum/16QAM/test/good
/mnt/data/wangbei/PromptAD/datasets/spectrum/16QAM/test/bad
```

### 协议

```text
k_shot = 1
训练时只取 train/good 中按文件排序后的前 1 张 normal 图像
测试集使用 test/good + test/bad
seed = 111
epochs = 50
```

如果新方法不依赖 epoch，也必须保持训练/测试划分一致。

---

## 4. 当前已完成结果（用于对照）

这个部分只用于告诉新 agent：当前任务难度大概在哪，哪些异常类型更稳定，哪些异常类型更难。它不要求新 agent 复现同一模型结构。

## 4.1 自测数据集当前结果

文件：

```text
analysis_outputs/universal_gray_residual_a01/method_comparison.csv
analysis_outputs/universal_gray_residual_a01/summary.md
analysis_outputs/universal_gray_residual_a01/current_vs_original_promptad.xlsx
```

结果：

| dataset | current reference result |
|---|---:|
| `burst_signal` | 90.0708 |
| `chirp_signal` | 84.9758 |
| `dsss_signal` | 96.6483 |
| **mean** | **90.5650** |

说明：
- 这是当前主线方法在自测主实验上的参考结果；
- 新方法至少要给出同协议下的三类结果和三类平均。

## 4.2 公开 RF 数据集当前结果

文件：

```text
analysis_outputs/public_current_vs_original/public_current_vs_original.csv
analysis_outputs/public_current_vs_original/public_current_vs_original.xlsx
analysis_outputs/public_current_vs_original/summary.md
analysis_outputs/public_gray_residual_a01/summary.md
```

当前协议结果：

| anomaly | level 1 reference | level 2 reference | level 3 reference | mean |
|---|---:|---:|---:|---:|
| `burst` | 96.88 | 88.47 | 65.29 | 83.55 |
| `chirp` | 94.16 | 71.75 | 55.55 | 73.82 |
| `dsss` | 94.99 | 84.40 | 71.88 | 83.76 |

档位对应：
- `burst`: `m30db / m40db / m50db`
- `chirp`: `m50db / m55db / m60db`
- `dsss`: `m20db / m30db / m40db`

说明：
- `burst` 和 `chirp` 在当前公开协议下更有区分度；
- `dsss` 仍然是最不稳定的一类；
- 新方法至少要给出三类分档结果、三类均值和九条总体均值。

## 4.3 spectrum 数据集当前结果

文件：

```text
analysis_outputs/spectrum_summary/spectrum_baseline_vs_current.csv
analysis_outputs/spectrum_summary/spectrum_baseline_vs_current.xlsx
analysis_outputs/spectrum_summary/spectrum_current_scheme_summary.xlsx
```

结果：

| class | current reference result |
|---|---:|
| `16QAM` | 98.48 |
| `CHIRP` | 98.48 |
| `GMSK` | 96.84 |
| `QPSK` | 94.10 |
| **average** | **96.975** |

---

## 5. 新 agent 必须遵守的执行规范

## 5.1 不要改实验协议

新 agent 不要自行改：
- 数据集路径；
- 训练/测试划分；
- 公开 RF 的混合难度协议；
- `seed = 111`；
- `epochs = 50`（如果方法本身不用 epoch，则至少要保证训练资源量合理且可说明）。

## 5.2 新方法实验必须隔离输出

不要覆盖现有结果目录。新方法必须使用独立目录：

```text
result_split_ablation/<new_method_name>_...
analysis_outputs/<new_method_name>_...
```

不要覆盖：

```text
analysis_outputs/universal_gray_residual_a01/*
analysis_outputs/public_current_vs_original/*
analysis_outputs/spectrum_summary/*
```

## 5.3 实验顺序建议

如果用户指定一种新的异常检测方法，推荐顺序：

1. **小范围 sanity check**
   - 先选 1~3 个代表性 case 跑通，确认代码和方法有效；
2. **自测主实验**
   - `burst_signal / chirp_signal / dsss_signal`
   - `normal_75_25`
   - 4 scenes × 3 noise levels；
3. **公开 RF 补充验证**
   - 使用当前固定混合协议；
4. **spectrum 补充验证**
   - 作为第三组补充结果。

## 5.4 资源约束

公开 RF 数据集较吃内存。

建议：
- 不要盲目大并发；
- 最多 2 并发；
- 如果内存吃紧，改成串行；
- 优先先跑代表性 case，再扩全量。

## 5.5 结果汇报格式

新 agent 至少需要产出：

1. 一个结果总表（csv 或 xlsx）；
2. 一个简短 `summary.md`；
3. 一段明确结论，回答：
   - 新方法在自测主实验上的三类结果和三类平均是多少；
   - 新方法在公开 RF 混合协议上的三类分档结果、三类均值和总体均值是多少；
   - 新方法在 `spectrum` 上的四类结果和平均值是多少；
   - 哪类异常改善，哪类异常退化；
   - 这个方法是否值得继续推进。

---

## 6. 新 agent 的任务模板

用户会另行告诉新 agent 要尝试的具体方法名称或思路。拿到方法后，该 agent 应按下面模板执行：

### 任务 1：方法接入

- 在独立目录中接入用户指定的新方法；
- 说明：
  - 方法原理是什么；
  - 为什么它可能对频谱异常检测有效；
  - 需要哪些输入、训练或采样设置；
  - 是否仍然满足“训练只用 normal、abnormal 仅测试”的约束。

### 任务 2：自测主实验

固定协议：

```text
自测三类异常
normal_75_25
4 scenes × 3 noise levels
seed = 111
epochs = 50
```

输出：
- `burst_signal / chirp_signal / dsss_signal` 的结果；
- 三类平均；
- 与当前参考结果的对比。

### 任务 3：公开 RF 补充验证

固定协议：
- `burst`: `m30/m40/m50`
- `chirp`: `m50/m55/m60`
- `dsss`: `m20/m30/m40`

输出：
- 三类分档结果；
- 各类均值；
- 九条总体均值；
- 与当前参考结果对比。

### 任务 4：spectrum 补充验证

输出：
- `16QAM / CHIRP / GMSK / QPSK` 的结果；
- 平均值；
- 与当前参考结果对比。

### 任务 5：结论判断

必须明确回答：

1. 这个新方法在自测主实验上是否值得继续；
2. 它在公开 RF 上是否对 `burst/chirp/dsss` 都有效，还是只对某一类有效；
3. 它在 `spectrum` 上是否稳定；
4. 它是否值得作为主线替代方案，还是只适合作为专项分支。

---

## 7. 当前关键文件位置

主文档：

```text
现有方案介绍.md
improve.md
```

自测主实验结果：

```text
analysis_outputs/universal_gray_residual_a01/method_comparison.csv
analysis_outputs/universal_gray_residual_a01/summary.md
analysis_outputs/universal_gray_residual_a01/current_vs_original_promptad.xlsx
```

公开 RF 结果：

```text
analysis_outputs/public_current_vs_original/public_current_vs_original.csv
analysis_outputs/public_current_vs_original/public_current_vs_original.xlsx
analysis_outputs/public_current_vs_original/summary.md
analysis_outputs/public_gray_residual_a01/summary.md
```

spectrum 结果：

```text
analysis_outputs/spectrum_summary/spectrum_baseline_vs_current.csv
analysis_outputs/spectrum_summary/spectrum_baseline_vs_current.xlsx
analysis_outputs/spectrum_summary/spectrum_current_scheme_summary.xlsx
```

---

## 8. 给新 agent 的一句话要求

不要改协议，不要覆盖现有结果，不要重复已经失败的老方向。用户指定新方法后，你的任务是：**在当前固定数据协议下，把这个新方法完整跑通，并清楚告诉用户它在自测数据、公开 RF 和 `spectrum` 三组实验上是否值得继续。**
