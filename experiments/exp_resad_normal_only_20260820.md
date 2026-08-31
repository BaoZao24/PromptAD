# ResAD normal-only baseline 实验记录

> **协议审计警告（2026-08-21）**：本记录中的原始 ResAD 结果不是论文正式主表结果。原实现为可训练的 ResAD 流头建立了源域模型，并将 In-house RF 的源域正常样本复用到了 Public RF、FedJam 和 OFDMA；这与项目要求的“各数据集独立、只使用该数据集允许的正常 support”不一致。因此，下面的数值只能保留为原始探索记录，不能用于跨方法公平排名。

日期：2026-08-20
方法：`ResAD-Normal-Only`  〔ResAD 机制的项目协议适配版〕

## 实验目的

将 ResAD 改为不使用异常样本和异常像素掩码的版本，作为更公平的外部 baseline。

## 原运行配置（已废弃）

- 源域训练只使用独立 In-house RF 场景中的正常频谱图；
- 不读取异常图像、异常标签或异常掩码；
- In-house RF 使用留一场景：每个目标场景分别用另外 3 个场景训练一个源模型；
- 每个源模型训练 100 个 epoch；
- 目标场景只使用正常 support 建立 ResAD 的参考特征库；
- 目标测试集不参与训练、校准或模型选择；
- 使用 Wide-ResNet-50-2 的 layer1、layer2、layer3 特征；
- 三个尺度分别进行匹配、残差建模和 Flow 评分，最后使用正常似然 `logp` 分支；
- 不报告未经过异常监督训练的 `bscore` 和 `merged` 分支；
- 指标为 AUROC、AUPRC 和 FPR@95%TPR。

## 正式项目实验设置（以 README 和论文实验部分为准）

四个数据集必须分别运行，不能共享源域训练数据、support、参考分布或测试统计量：

- **In-house RF**：按目标场景独立建立正常 support；每个频点使用 1 个正常样本。support 只服务于同一目标场景的测试单元，不跨场景复用。
- **Public RF**：只使用 Public RF 自己的正常 support，按每频点 1/2/4-shot 分别实验；测试集固定为 Public RF 的独立 full test。
- **FedJam**：只使用 FedJam 自己的 benign/正常训练样本，按 1/2/4-shot 建立 support；在 FedJam 自己的 full test 上评估，不使用其他 RF 数据集的样本或 KPI 信息。
- **OFDMA**：每个目标场景独立使用该场景的 1/2/4 个正常观测建立 support，并在同一场景的独立测试数据上评估；不把 In-house RF 作为 OFDMA 的训练源域。

原脚本中的 `rf_source_all_scenes` 配置不符合正式协议：Public RF、FedJam、OFDMA 的旧结果需要重新按各自数据集独立适配；In-house RF 的旧 leave-one-scene-out 结果也只能作为额外探索，不能冒充项目主线的 target-only 结果。

## 修正后的实验代码（2026-08-21）

`tools/eval_resad_formal.py` 已改为 `target-support-only-v2`：

- In-house RF 按目标场景分别拟合；
- Public RF 按 `k=1/2/4-per-frequency` 分别拟合；
- FedJam 按 1/2/4-shot 分别拟合；
- OFDMA 按 `目标场景 × shot` 分别拟合；
- ResAD 的可训练头、正常分数校准和推理参考库都只使用当前单元允许的正常 support；
- 不读取其他数据集、异常图、异常标签、像素掩码或测试集统计量；
- 拟合时将 support 确定性分成两组，互相作为正常参考，避免同一图像匹配自身产生全零残差。只有真正的单图 support 才使用一个固定的轻微噪声视图作为拟合输入；原始 support 仍用于推理参考库。

脚本会把每个 support 的签名、数据集范围、拟合方式和样本数写入 checkpoint 与 `results.json`。加载 checkpoint 时如发现数据集、场景、shot 或 support 签名不一致，会直接报错。

四套协议的 `--validate-only` 检查均已通过；Public RF、FedJam 1-shot 和 OFDMA 单场景 1-shot 的缩小输入、1 epoch CPU 冒烟测试也已完整跑通拟合、建库、评分和结果写出。正式结果尚未重跑。

## In-house RF 结果

共完成 4 个留一场景源模型和 60 个测试单元，所有指标均为有限值。

| 方法 | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|
| ResAD-Normal-Only | 80.6799 | 65.8164 | 65.7923 |

结果说明：该方法仍属于“独立源域正常样本预训练” baseline，不是完全免训练方法；但没有使用任何异常监督或像素掩码。

## 输出文件

- 指标汇总：[metrics.csv](../analysis_outputs/resad_normal_only_rf_target/metrics.csv)
- 完整运行信息：[results.json](../analysis_outputs/resad_normal_only_rf_target/results.json)
- 源模型与逐单元分数：`analysis_outputs/resad_normal_only_rf_target/`
- 适配代码：[eval_resad_formal.py](../tools/eval_resad_formal.py)

## Public RF 结果

Public RF 使用固定的 k-per-frequency support 和完整固定测试集，共 15 个测试单元；三个 shot 使用同一测试集合。

| 方法 | k | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|---:|
| ResAD-Normal-Only | 1 | 69.6301 | 16.7955 | 69.5195 |
| ResAD-Normal-Only | 2 | 72.1223 | 19.2874 | 63.1276 |
| ResAD-Normal-Only | 4 | 75.7172 | 25.1580 | 59.0156 |

结果文件：`analysis_outputs/resad_normal_only_public_rf/metrics.csv`。

## FedJam 结果

FedJam 使用官方测试集的 1/2/4-shot 冷启动协议；每个 shot 的测试集合固定为 1,800 张正常图和 5,400 张异常图，共 7,200 张测试图。源域训练仍只使用 In-house RF 正常频谱图，不使用 FedJam 异常样本或异常掩码。

| 方法 | k | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|---:|
| ResAD-Normal-Only | 1 | 60.4331 | 81.6668 | 88.5000 |
| ResAD-Normal-Only | 2 | 63.5136 | 83.3712 | 90.2222 |
| ResAD-Normal-Only | 4 | 64.4998 | 83.6502 | 89.2778 |

结果文件：`analysis_outputs/resad_normal_only_fedjam/metrics.csv`。

## OFDMA 结果

OFDMA 使用 30 个官方测试场景的完整测试集。每个场景包含 100 个正常观测和 100 个异常观测，每个观测由 21 个感知单元频谱图组成；因此每个 shot 的测试协议固定不变，support 分别为 1/2/4 个正常观测（21/42/84 张 support 图）。共生成 90 个场景-shot 结果文件。

| 方法 | k | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|---:|
| ResAD-Normal-Only | 1 | 55.2357 | 58.4652 | 92.9000 |
| ResAD-Normal-Only | 2 | 62.2530 | 65.9493 | 89.7667 |
| ResAD-Normal-Only | 4 | 68.9487 | 72.7392 | 83.0333 |

结果文件：[metrics.csv](../analysis_outputs/resad_normal_only_ofdma/metrics.csv)；场景级分数：`analysis_outputs/resad_normal_only_ofdma/scores/`。

## 结果分析：协议审计结论

此前文档把结果差异归因于“ResAD 从 In-house RF 跨域迁移到其他数据集”。这个判断虽然描述了代码当时的运行方式，但不能作为正式实验结论，因为跨数据集源域复用本身已经违反了项目实验设置。当前结果只能说明“这次跨域配置下的 ResAD 表现”，不能说明 ResAD 在各目标数据集独立协议下的真实能力。

正式分析应在四个数据集分别重跑后进行，并至少区分：

1. 数据集自己的正常 support 是否足以稳定建立 ResAD 所需的统计/流模型；
2. 同一数据集内部不同场景或频点的正常变化是否被误报；
3. ResAD 的训练需求与本项目 target-only、免训练主线之间的协议差异。

## 后续实验

当前结果已作为“原始跨域配置审计记录”保留。四个数据集的正式 ResAD-Normal-Only 结果必须按上述独立协议重新运行后，才能进入 baseline 主表或用于论文结论。
