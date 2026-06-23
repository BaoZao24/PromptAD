# PromptAD — RF 频谱图少样本异常检测

> [English](./README.md) | **中文**

本仓库基于 [PromptAD (CVPR 2024)](http://arxiv.org/abs/2404.05231) 改造而来，用于 **射频频谱图（RF spectrogram）的少样本异常检测**。
模型仅在“正常（无干扰）频谱图块”上训练，测试时识别 `burst / chirp / dsss / wideband_pulse` 等干扰信号引入的异常。

> 上游代码保留了 MVTec / VisA 接口；本仓库的主要改动集中在：RF prompt 适配、频谱多通道输入构造、score 融合、跨库迁移、以及多种轻量适配模块（VCPA / Visual Adapter / Visual LoRA / RN50 引导）。

---

## 1. 主方案与默认实验配置

按照当前实验记录，主方案为：

```text
prompt_mode      = rf                                # 频谱场景化 prompt 模板
input_mode       = morph_fusion_gray_residual_a01    # gray_contrast + weak_residual(α=0.1) + original_gray
cls_score_mode   = text_only                         # 仅使用文本异常分数
split_mode       = normal_75_25                      # 3/4 normal 训练，1/4 normal + 全部 abnormal 测试
seed             = 111
k-shot           = 1
Epoch            = 50
backbone         = ViT-B-16-plus-240 (laion400m_e32)
```

默认对比对象（不含 wideband_pulse）：

| 信号类型 | 噪声等级 (JSR) |
|---|---|
| `burst_signal`、`chirp_signal`、`dsss_signal` | `m10db`、`m20db`、`m30db` |
| `wideband_pulse` | `m20db`、`m30db`、`m40db` |

四个测试场景固定为：`WeaponMuseum_spectrum`、`Playground_spectrum`、`TimeSquare_spectrum`、`Gymnasium_spectrum`。

更详细的方案动机和图示见 [`现有方案介绍.md`](./现有方案介绍.md)、[`improve.md`](./improve.md)、[`提示词机制说明.md`](./提示词机制说明.md)、[`分数机制说明.md`](./分数机制说明.md)。

---

## 2. 安装

```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

主要依赖：PyTorch (CUDA 11.8)、`open_clip_torch`、`timm`、`transformers`、`opencv-python`、`scikit-learn`、`pandas`、`loguru`、`tqdm`。

---

## 3. 数据组织

```
/mnt/data/wangbei/data/datasets/
├── normal/                                # 正常（无干扰）训练块；所有信号类型共用
│   ├── WeaponMuseum_spectrum/
│   ├── Playground_spectrum/
│   ├── TimeSquare_spectrum/
│   └── Gymnasium_spectrum/
├── burst/                                 # burst 干扰测试集
│   └── {scene}/
│       ├── normal/{noise_level}/          # 干净测试块
│       ├── abnormal/{noise_level}/        # 异常块
│       └── groundtruth/{noise_level}/     # 像素级 GT mask
├── chirp/                                 # chirp 干扰（结构相同）
├── dsss/                                  # DSSS 干扰（结构相同）
├── wideband/                              # 宽带脉冲（结构相同，噪声等级不同）
└── deceptive/                             # deceptive（stealthy）干扰，仅 0db
```

数据集 loader 集中在 [`datasets/`](./datasets/)，统一通过 `datasets.__init__.get_dataloader_from_args(...)` 构造。
切分由 `--split-mode` 控制：

- `legacy`：原始 PromptAD few-shot 切分；
- `normal_75_25`：本仓库默认，3/4 正常训练 + 1/4 正常和全部异常测试，避免训练/测试泄漏。

---

## 4. 项目结构

```
PromptAD/
├── train_cls.py              # 单次图像级训练 / 评估入口（CLS 任务）
├── train_seg.py              # 像素级（SEG）训练入口
├── test_cls.py / test_seg.py # 仅评估（不重新训练）
├── run_cls.py / run_seg.py   # 早期 batch 脚本（mvtec/visa/spectrum/sample）
├── run_rf_split_all.py       # 主 batch 脚本：3 信号 × 4 场景 × 3 噪声 × N GPU 并行
├── plot_scoremap.py          # 输入 / GT / score map 三联对比可视化
│
├── PromptAD/
│   ├── ad_prompts.py         # Prompt 模板（generic / rf / rf_domain / rf_signal_structured ...）
│   ├── model.py              # 全部核心模型类（详见 §5）
│   └── CLIPAD/               # 适配后的 OpenCLIP 视觉/文本编码器
│
├── datasets/
│   ├── __init__.py           # dataloader factory + 主入口 get_dataloader_from_args
│   ├── dataset.py            # CLIPDataset 基类（公共加载/裁剪逻辑）
│   ├── burst_signal.py       # burst loader（含 normal_75_25 切分）
│   ├── chirp_signal.py       # chirp loader
│   ├── dsss_signal.py        # dsss loader
│   ├── wideband_pulse.py     # wideband loader（PNG / NPY 两套）
│   ├── deceptive_signal.py   # deceptive 干扰
│   ├── rf_spe_png.py         # RF 通用 PNG 数据集
│   ├── spectrum.py / sample.py
│   ├── rf_split_utils.py     # normal_75_25 切分工具
│   ├── seeds_mvtec/, seeds_visa/  # 上游 few-shot 种子
│   └── mvtec.py / visa.py / prepare_visa_public.py
│
├── utils/
│   ├── training_utils.py     # 输出目录、setup_seed、TripletLoss 等
│   ├── csv_utils.py          # 结果 CSV 写入
│   ├── metrics.py            # AUROC / AUPRO / metric_cal_img(harmonic 融合)
│   ├── eval_utils.py         # specify_resolution 等评估工具
│   └── visualization.py      # plot_sample_cv2 score map 输出
│
├── tools/                    # 离线分析与可视化脚本（见 §8）
├── experiments/              # 已沉淀的实验：baseline 重定义、band-aware、normal_bg、VAE 融合 …
├── result/                   # 实验输出根目录（运行后自动生成）
├── archive_results/          # 历史结果归档
│
├── 现有方案介绍.md            # 主方案 & 故事线
├── improve.md                # 阶段性改进报告
├── 提示词机制说明.md          # PromptLearner 工作流
├── 分数机制说明.md            # 分数融合机制
└── 跨库训练.md                # 跨库迁移说明（参考 AdaptCLIP）
```

---

## 5. 核心模型组件

文件 [`PromptAD/model.py`](./PromptAD/model.py) 较大（约 2.8k 行），按功能可分为三层：

### 5.1 输入通道构造（频谱多通道融合）

针对 RF 频谱图“频域结构 + 弱残差能量”的特性，提供多套三通道融合策略：

| 类名 | 含义 |
|---|---|
| `MorphFusionGrayResidualChannels` (`morph_fusion_gray_residual_a01`) | **主方案**：gray_contrast + weak_residual(α) + original_gray |
| `MorphFusionGrayResEnergyChannels` / `MorphFusionGrayResBandChannels` | 用 energy / 频带统计替换 weak_residual |
| `MorphFusionGaborResidualChannels` 系列 | Gabor 方向滤波 + 残差 |
| `MorphFusionPlusChannels` / `MorphFusionDualGradChannels` / `MorphFusionBalancedChannels` | 早期消融变体 |
| `DSSSWeakResidualChannels` / `DSSSEnergyProfileChannels` 等 | 仅 DSSS 用的窄带弱信号增强 |
| `ChirpDirectionalChannels` / `ChirpRidgeChannels` / `ChirpTrackEnhanceChannels` | chirp 时频脊增强 |
| `NormalBgDeviationChannels` | normal/bg 偏离归一化（用 `'normal_bg' in input_mode` 触发） |
| `LogPowerChannels` | 单通道 log-power 压缩 |

此外 `--input-mode auto` 会按 dataset 自动选择默认输入；`--input-mode signal_adaptive*` 会按异常类型动态切换。

### 5.2 Prompt 与文本分支

- `PromptLearner`：CoOp 风格的可学习 prompt
  - 正常 prompt：可学习 context + 类别名；
  - 异常 prompt（handle）：使用 `rf_state_anomaly` 等模板；
  - 异常 prompt（learned）：可学习 context + 异常前缀 + 类别名。
- Prompt 模板池由 `--prompt-mode` 选择：`generic / rf_domain / rf / legacy / rf_object_agnostic / rf_scene_conditioned / rf_signal_structured`。
- 文本原型聚合方式：`--text-prototype-mode {single, grouped_max, grouped_mean, grouped_meanmax, grouped_softmax}`。

### 5.3 视觉分支与异常分数

PromptAD 主类整合：

1. **CLIP 视觉编码器**（默认 ViT-B-16-plus-240）→ patch token + cls token；
2. **特征 Gallery**：用全部正常训练样本构造 `feature_gallery1/2`、`global_features`，用于视觉距离分数；
3. **Textual anomaly score**：cls 特征 vs 正常/异常文本原型相似度差；
4. **Visual anomaly map**：patch token 与 normal gallery 的最近邻距离图；
5. **图像级融合**：由 `--cls-score-mode` 决定（`text_only / visual_topk / visual_topk_max / visual_topk_freq / normal_center / normal_mahalanobis / text_normal_center / text_normal_mahalanobis`），权重为 `α / β / γ`。

可选的扩展模块（默认全部关闭，按 flag 启用）：

| Flag | 含义 |
|---|---|
| `--visual-adapter` | `ResidualVisualAdapter`：冻结 CLIP 后接残差 bottleneck，仅训练 adapter |
| `--visual-lora` | 在 CLIP 视觉 transformer attention 上加 LoRA |
| `--visual-class-prompt` (VCPA) | `VisualClassPromptAdapter`：把正常视觉原型映射成软 class token，注入文本侧 |
| `--learnable-score-fusion` | `LearnableScoreFusionHead`：在文本分数上做 BCE 残差校正 |
| `--rn50-visual-fusion` | 冻结 CLIP-RN50 分支：global / local_topk 两种打分；`--rn50-fusion-mode guided_vit` 时用 RN50 局部图引导 ViT patch 聚合 |
| `--cnn-vit-mamba-fusion` | `CNNMambaLocalBranch`：CNN + ViT + Tiny-Mamba 局部分支，可作为辅助分数和正常对齐损失 |
| `--multiview-fusion` | 评估时对 `(rgb, spectral_gradient_v2, dsss_weak_residual)` 三视图做 max/mean/conservative 在线融合 |
| `--stat-fusion` | 直接把传统频谱统计 top-k 分数线性融合到模型分数中 |

> 这些扩展实现的入口都在 `model.py` 中，例如 `ResidualVisualAdapter:1306`、`VisualClassPromptAdapter:1325`、`LearnableScoreFusionHead:1351`、`CNNMambaLocalBranch:1420`、`PromptLearner:1438`、`PromptAD:1698`。

---

## 6. 单次训练 / 评估

```bash
python train_cls.py \
    --dataset burst_signal \
    --class_name Playground_spectrum \
    --noise-level m10db \
    --k-shot 1 --Epoch 50 --gpu-id 0 \
    --prompt-mode rf \
    --input-mode morph_fusion_gray_residual_a01 \
    --cls-score-mode text_only \
    --split-mode normal_75_25 --normal-train-ratio 0.75 \
    --seed 111 --vis True
```

常用参数（完整定义见 `train_cls.py:get_args`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset` | `mvtec` | `burst_signal / chirp_signal / dsss_signal / wideband_pulse / wideband_pulse_png / rf_spe_png / mvtec / visa / spectrum / sample / deceptive_signal` |
| `--class_name` | `carpet` | 测试场景名（频谱任务下为 `*_spectrum`） |
| `--noise-level` | `m10db` | JSR 等级 |
| `--k-shot` | `1` | 正常样本数（normal_75_25 模式下意义改为参与梯度更新的正常样本数） |
| `--Epoch` | `50` | 训练轮数 |
| `--backbone` | `ViT-B-16-plus-240` | CLIP backbone（可选 `ViT-B-16`） |
| `--seed` | `111` | 随机种子 |
| `--vis` | `False` | 是否保存 score map 可视化 |
| `--prompt-mode` | `rf` | 文本 prompt 模板池 |
| `--input-mode` | `auto` | 频谱多通道输入模式 |
| `--cls-score-mode` | `text_only` | 图像级分数融合方式 |
| `--split-mode` | `legacy` | `legacy / normal_75_25` |
| `--root-dir` | `./result` | 输出根目录 |

> 评估指标使用 `utils.metrics.metric_cal_img`，对 image-level 分数与 anomaly map 的 max 做调和融合后再算 AUROC。详见 [`分数机制说明.md`](./分数机制说明.md)。

---

## 7. 批量运行（主方案推荐入口）

```bash
# 默认运行 3 个信号 × 4 个场景 × 3 个噪声等级，自动检测可用 GPU 并行
python run_rf_split_all.py \
    --epochs 50 \
    --prompt-mode rf \
    --input-mode morph_fusion_gray_residual_a01 \
    --cls-score-mode text_only

# 指定 GPU、子集和参数
python run_rf_split_all.py --gpus 0 1 2 3 \
    --datasets burst_signal chirp_signal dsss_signal \
    --epochs 50 --seed 111

# 干跑确认命令
python run_rf_split_all.py --dry-run
```

脚本会：
1. 跳过已经写过 `Seed_111-results.csv` 且 `i_roc > 0` 的 job（除非 `--force`）；
2. 把每个 job 的日志存到 `/tmp/rf_split_<dataset>_<scene>_<noise>_gpu<gid>.log`；
3. 全部跑完后按 `dataset` 打印 4×3 的 `i_roc` 汇总表。

RF 跨库 Adapter 实验计划见 [`docs/PromptAD_adapter_experiment_plan.md`](./docs/PromptAD_adapter_experiment_plan.md)。正式 PromptAD RF 跨库入口还需要后续实现。

---

## 8. 输出目录与可视化

```
result/<dataset>/<scene>/<noise>/<split>/k_<shot>/
├── csv/Seed_<seed>-results.csv      # i_roc / p_roc / pro 等指标
├── checkpoint/                      # PromptLearner / Adapter / Gallery 状态
├── scores/Seed_<seed>-image_scores.npz   # names / scores / labels / visual_maps / text_scores
└── imgs/                            # plot_sample_cv2 输出（启用 --vis True 时）
```

可视化对比工具：

```bash
# 默认遍历 ./result，每种 (dataset/scene/noise) 抽 1 张
python plot_scoremap.py --root-dir ./result --out ./result/scoremap_compare.png

# 过滤
python plot_scoremap.py --dataset burst_signal --scene Playground_spectrum --noise m10db
```

`tools/` 下还有大量离线分析脚本（PR 曲线、特征图导出、消融对比等），常用的有：

- `tools/export_morph_fusion_featuremaps.py` — morph_fusion 系列的输入特征对比；
- `tools/plot_rf_ablation_heatmap.py` — RF 消融热力图；
- `tools/extract_normal_maps.py` — 正常样本的 anomaly map 抽样。

---

## 9. 实验沉淀

- [`experiments/baseline_redefinition/`](./experiments/baseline_redefinition/) — Original PromptAD（`legacy` prompt + RGB）vs 当前 RF 方案；澄清“RGB baseline ≠ original PromptAD”。
- [`experiments/failed_directions_summary.md`](./experiments/failed_directions_summary.md) — 已验证不奏效的方向汇总。
- [`docs/PromptAD_adapter_experiment_plan.md`](./docs/PromptAD_adapter_experiment_plan.md) — RF 跨库 Adapter 实验计划。

> 内部约定：所有新方法都需要在主方案（`rf` + `morph_fusion_gray_residual_a01` + `text_only` + `normal_75_25`，seed=111，k=1）下与原方案对比，不假设测试时已知异常类型。

---

## 10. 上游工作 & 引用

```bibtex
@article{li2024promptad,
  title={PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection},
  author={Li, Xiaofan and Zhang, Zhizhong and Tan, Xin and Chen, Chengwei and Qu, Yanyun and Xie, Yuan and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2404.05231},
  year={2024}
}
```

致谢：[WinCLIP](https://github.com/caoyunkang/WinClip.git)、[CoOp](https://github.com/KaiyangZhou/CoOp.git)、[OpenCLIP](https://github.com/mlfoundations/open_clip)。
