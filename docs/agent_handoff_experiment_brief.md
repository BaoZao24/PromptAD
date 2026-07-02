# Agent Handoff: Current Main Scheme Is RF + RGB

更新时间：2026-06-26
项目根目录：`/mnt/data/wangbei/PromptAD`

> **归档说明（2026-07-02）**：本文是数据集更新初期的历史交接记录，不再作为当前实验执行依据。
> 当前统一实验协议以 `docs/agent_handoff_method_funnel_overview.md` 为准。

## 1. 当前状态

数据集已更新。旧 `result/` 和旧 `analysis_outputs/` 里的结果全部视为失效，不要再作为结论引用。

第一阶段 baseline 重测已经完成，结果目录：

```text
analysis_outputs/20260626_dataset_update_rerun/
```

核心产出：

```text
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/baseline_compare_long.csv
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/baseline_compare_summary.csv
analysis_outputs/20260626_dataset_update_rerun/03_summaries/main_findings.md
```

本 handoff 的当前决策是：

**先采用 `rf + rgb` 作为当前主线方案。**

原因很简单：

- `rf + rgb` 和 `legacy + rgb` 几乎持平，overall 只低 `0.0367`。
- `rf + morph_fusion_gray_residual_a01` 有明显负效果。
- `rf + gray_local2d_edge` 虽然提升 burst，但明显伤害 chirp 和 pulse，不是通用方案。
- 因此先不要继续把主线押在手工输入变换上，当前主线回到最稳的 RGB 输入。

不要重跑已完成实验，除非发现文件缺失或结果损坏。

## 2. 固定实验协议

任务类型：频谱图异常检测。
主指标：`Image-AUROC`，也就是结果表里的 `i_roc`。

训练材料：

- 每个异常类型、每个场景、每个 JSR 档位内部，只用 normal 样本训练。
- abnormal 样本只用于测试。
- normal 样本按固定比例切分：75% normal train，25% normal test。

测试材料：

- 测试集 = 剩余 25% normal + 当前 JSR 下全部 abnormal。

固定参数：

```text
split_mode = normal_75_25
normal_train_ratio = 0.75
k_shot = 1
seed = 111
epochs = 50
cls_score_mode = text_only
batch_size = 400
vis = False
```

注意：`k_shot=1` 是历史代码路径保留参数；在 `normal_75_25` 模式下，实际训练口径以 `normal_train_ratio=0.75` 为准。

## 3. 数据集范围

只跑以下三个异常类型：

```text
burst_signal
chirp_signal
pulse_signal
```

不要跑：

```text
dsss_signal
wideband_pulse
rf_spe_png
rf_public_pooled_smoke
rf_target_test_pool
spectrum
```

场景固定为：

```text
WeaponMuseum_spectrum
Playground_spectrum
TimeSquare_spectrum
Gymnasium_spectrum
```

JSR 档位固定为：

```text
burst_signal: m10db, m20db, m30db
chirp_signal: m10db, m20db, m30db
pulse_signal: m20db, m30db, m40db
```

当前代码中 `run_rf_split_all.py` 已经把 `pulse_signal` 设置为 `m20db/m30db/m40db`。

## 4. 已完成结果

已完成四组：

```text
legacy_rgb
rf_rgb
rf_morph_a01
rf_gray_local2d_edge
```

每组 36 个 job，四组合计 144 个 job。

平均 Image-AUROC：

| method | burst_signal | chirp_signal | pulse_signal | overall |
|---|---:|---:|---:|---:|
| `legacy + rgb` | 96.8083 | 96.1033 | 91.6950 | **94.8689** |
| `rf + rgb` | 96.8225 | 95.9792 | 91.6950 | **94.8322** |
| `rf + morph_fusion_gray_residual_a01` | 94.7075 | 95.4108 | 90.1967 | 93.4383 |
| `rf + gray_local2d_edge` | 97.3392 | 92.9033 | 90.5800 | 93.6075 |

相对 `legacy + rgb`：

| method | burst Δ | chirp Δ | pulse Δ | overall Δ |
|---|---:|---:|---:|---:|
| `rf + rgb` | +0.0142 | -0.1242 | +0.0000 | -0.0367 |
| `rf + morph_fusion_gray_residual_a01` | -2.1008 | -0.6925 | -1.4983 | -1.4306 |
| `rf + gray_local2d_edge` | +0.5308 | -3.2000 | -1.1150 | -1.2614 |

结论：

- 数值上最强 baseline 是 `legacy + rgb`，overall = `94.8689`。
- 项目当前决定先采用 `rf + rgb` 作为主线，overall = `94.8322`，与 `legacy + rgb` 几乎持平。
- `rf + morph_fusion_gray_residual_a01` 明显低于 `rf + rgb` 和 `legacy + rgb`。
- `rf + gray_local2d_edge` 对 burst 有提升，但 chirp 下降 `3.20`，pulse 下降 `1.12`，不是通用方案。
- 当前不继续把 `morph` 或 `gray_local2d_edge` 作为主线输入。

重要附注：

- `pulse_signal` 上 `rf_rgb` 和 `legacy_rgb` 完全相同，是因为当前 `PromptAD/ad_prompts.py` 的 `get_rf_signal_key` 没识别 `pulse`，所以 RF prompt 对 pulse 没起作用。
- 如果后续要研究 RF prompt 本身，应单独补 pulse prompt；但本轮下一步只做输入方法，不改 prompt。

## 5. 当前采用方案

当前主线：

```text
prompt_mode = rf
input_mode = rgb
cls_score_mode = text_only
```

复现实验命令：

```bash
python run_rf_split_all.py \
  --datasets burst_signal chirp_signal pulse_signal \
  --gpus 0 1 2 3 \
  --epochs 50 \
  --seed 111 \
  --root-dir analysis_outputs/20260626_dataset_update_rerun/runs/rf_rgb \
  --prompt-mode rf \
  --input-mode rgb \
  --cls-score-mode text_only \
  --batch-size 400 \
  --force
```

注意：

- 虽然 `legacy + rgb` 数值略高，但差距只有 `0.0367`，当前项目决策优先保留 RF prompt 主线。
- 后续如果要追求最高单表数值，应同时报告 `legacy + rgb` 作为最强数值 baseline。

## 6. 已实现但暂不采用的输入方法

已经实现：

```text
--input-mode gray3
--input-mode gray_local2d_edge
```

`gray3`：

```text
C1 = gray
C2 = gray
C3 = gray
```

`gray_local2d_edge`：

```text
C1 = gray 原始黑白频谱
C2 = local2d residual 局部二维残差
C3 = edge gradient 边缘梯度
```

相关代码：

```text
PromptAD/model.py
train_cls.py
run_rf_split_all.py
tools/export_input_candidate_gallery.py
```

效果图：

```text
analysis_outputs/01_figures/input_candidates_20260626/candidate_composites.png
analysis_outputs/01_figures/input_candidates_20260626/candidate_middle_channels.png
analysis_outputs/01_figures/input_candidates_20260626/candidate_triplets.png
analysis_outputs/01_figures/input_candidates_20260626/README.md
```

已做 smoke 检查：

```text
python -m py_compile tools/export_input_candidate_gallery.py PromptAD/model.py train_cls.py run_rf_split_all.py
python run_rf_split_all.py --datasets pulse_signal --input-mode gray_local2d_edge --epochs 1 --gpus 0 --dry-run
```

`gray_local2d_edge` 已完成全量实验：

```text
rf + gray_local2d_edge overall = 93.6075
vs rf + rgb: -1.2247
vs legacy + rgb: -1.2614
```

结论：暂不采用。

## 7. 暂停的候选队列

以下候选先不要继续跑，除非用户重新要求探索输入变换：

```text
rf + gray3
rf + morph_fusion_clahe_gray
rf + morph_fusion_tophat_a01
rf + morph_fusion_multiscale_residual_a01
rf + morph_fusion_local2d_residual_a01
```

建议命名：

```text
runs/rf_gray3
runs/rf_clahe_gray
runs/rf_tophat_a01
runs/rf_multiscale_residual_a01
runs/rf_local2d_residual_a01
```

对应命令只需要替换：

```text
--root-dir analysis_outputs/20260626_dataset_update_rerun/runs/<method_name>
--input-mode <input_mode>
```

## 8. 结果文件

当前结果已经整理到：

```text
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/baseline_compare_long.csv
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/baseline_compare_summary.csv
analysis_outputs/20260626_dataset_update_rerun/03_summaries/main_findings.md
```

当前 `aggregate_results.py` 已能聚合四个方法：

```text
legacy_rgb
rf_rgb
rf_morph_a01
rf_gray_local2d_edge
```

## 9. 数据集更新后的 VCP 支线 Seg 任务计划

当前项目决策：

```text
Image-AUROC 主线先采用 rf + rgb。
Seg / Pixel-AUROC 可以单独评估 VCP 支线。
```

这两个任务不要混在一起：

- image-level 主表继续看 `i_roc`，当前主线是 `rf + rgb`。
- seg 支线只看 `p_roc`，目标是让 anomaly map 更准。
- VCP seg 结果不能拿来宣称 image ROC 提升，除非另做 image-level 对照。

### 9.1 已有 seg baseline 现状

已有原始 PromptAD seg 路径结果：

```text
analysis_outputs/20260626_dataset_update_rerun/runs_seg/legacy_rgb/
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/seg_compare_long.csv
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/seg_compare_summary.csv
analysis_outputs/20260626_dataset_update_rerun/03_summaries/seg_findings.md
```

当前原始 seg baseline：

| method | burst_signal | chirp_signal | pulse_signal | overall |
|---|---:|---:|---:|---:|
| `legacy + rgb` | 98.4958 | 96.9617 | 98.1645 | 97.8657 |

注意：

- 这是 PromptAD 原始 `forward(task='seg')`，没有 VCP / dense / text-aligned dense 支线。
- 这些 seg 结果对应的是**旧训练样本版本**。
- 既然 VCP 支线的训练样本已经调整，之前的 seg baseline 和任何 VCP seg 结果都不应继续使用。
- 下一步不是补 1 个缺失项，而是按新训练样本**整套重训** seg baseline，再做 VCP 对照。

### 9.2 先按新训练样本重训原始 seg baseline

原始 seg baseline 仍然用：

```text
prompt = legacy
input = rgb
```

但要基于**更新后的训练样本**重新跑完整 36 个 job，不要复用旧 `runs_seg/legacy_rgb/`。

建议新目录：

```text
analysis_outputs/20260626_dataset_update_rerun/runs_seg_refresh/legacy_rgb/
analysis_outputs/20260626_dataset_update_rerun/00_logs/seg_refresh_legacy_rgb/
```

建议命令：

```bash
python run_rf_split_all_seg.py \
  --gpus 0 1 2 3 \
  --epochs 50 \
  --seed 111 \
  --root-dir analysis_outputs/20260626_dataset_update_rerun/runs_seg_refresh/legacy_rgb \
  --log-dir analysis_outputs/20260626_dataset_update_rerun/00_logs/seg_refresh_legacy_rgb \
  --prompt-mode legacy \
  --input-mode rgb \
  --batch-size 400 \
  --force
```

然后把 `aggregate_seg_results.py` 指到新的 seg root 聚合。

如果要保留旧结果，只能当 archive，不能与这轮新训练样本结果混表。

### 9.2 补充说明：为什么必须整套重训

因为 seg 路径对训练 normal 的 feature gallery 很敏感：

```text
train_seg.py 会先从 train normal 建 image feature gallery
然后再训练 prompt learner
最后在 test 上评估 p_roc
```

训练样本一变，至少这些都会变：

```text
feature_gallery1
feature_gallery2
text feature 的最优 epoch
最终 anomaly map
```

所以不能只补单个场景或单个 JSR，必须整套重训。

### 9.2 重新聚合 seg baseline

```bash
python analysis_outputs/20260626_dataset_update_rerun/aggregate_seg_results.py \
  --root analysis_outputs/20260626_dataset_update_rerun
```

注意：如果聚合脚本里 `runs_dir = root / "runs_seg"` 写死了，而你把新结果放到 `runs_seg_refresh/`，
就需要先把聚合脚本改成接受 `--runs-subdir`，或者临时复制一个 `aggregate_seg_results_refresh.py`。

### 9.3 VCP seg 的实现前置条件

不要直接用当前 `run_rf_split_all_seg.py` 宣称跑了 VCP。

原因：

- `run_rf_split_all_seg.py` 现在只把 `prompt_mode/input_mode/batch_size` 传给 `train_seg.py`。
- `train_seg.py` 目前只训练 prompt learner，没有接 `--visual-class-prompt`、`--text-aligned-dense`、`--dense-mask-branch` 等 VCP/dense 参数。
- 所以当前 seg 入口只能代表原始 PromptAD seg，不能代表 VCP 支线。

先做最小实现：

```text
目标脚本：train_seg.py + run_rf_split_all_seg.py
保留默认行为不变。
新增可选 VCP 参数，默认 False。
```

需要给 `train_seg.py` 补的参数：

```text
--visual-class-prompt {True,False}
--visual-class-token-num
--visual-class-prompt-bottleneck-ratio
--visual-class-prompt-alpha
--visual-class-prototype-mode {mean,diverse}
--visual-class-prototype-num
--visual-class-prompt-lr
```

需要在 `train_seg.py` 的 feature gallery 阶段补：

```text
1. 除 features1/features2 外，同时收集 global_features。
2. build_image_feature_gallery(features1, features2, global_features)。
3. 如果启用 visual_class_prompt，调用 model.set_visual_class_prototype(global_features)。
4. optimizer 使用 model.get_trainable_parameters_with_lrs(...)，而不是只优化 prompt_learner.parameters()。
```

需要给 `run_rf_split_all_seg.py` 补：

```text
--visual-class-prompt
--visual-class-token-num
--visual-class-prompt-alpha
--visual-class-prototype-mode
--visual-class-prototype-num
--visual-class-prompt-lr
```

并把这些参数传给 `train_seg.py`。

### 9.4 VCP seg 第一组实验

第一组只跑 VCPA，不上 dense mask，不上 LoRA，不上 visual adapter。

方法名：

```text
rf_vcpa_rgb
```

建议命令，前提是 9.3 的参数已经接好：

```bash
python run_rf_split_all_seg.py \
  --gpus 0 1 2 3 \
  --epochs 50 \
  --seed 111 \
  --root-dir analysis_outputs/20260626_dataset_update_rerun/runs_seg_refresh/rf_vcpa_rgb \
  --log-dir analysis_outputs/20260626_dataset_update_rerun/00_logs/seg_refresh_rf_vcpa_rgb \
  --prompt-mode rf \
  --input-mode rgb \
  --batch-size 400 \
  --visual-class-prompt True \
  --visual-class-token-num 2 \
  --visual-class-prompt-alpha 0.2 \
  --visual-class-prototype-mode mean \
  --force
```

判断标准：

- 主指标：`p_roc`。
- 对照 1：**新训练样本版本**的 `legacy_rgb` seg baseline。
- 对照 2：如果时间允许，也跑一个 `rf_rgb` seg baseline，确认 RF prompt 本身对 seg map 的影响。
- 如果 `rf_vcpa_rgb` 只提升某一类、但伤害其他类，不能算通用 seg 改进。

### 9.5 Seg 聚合脚本需要更新

当前：

```text
analysis_outputs/20260626_dataset_update_rerun/aggregate_seg_results.py
```

里面 `METHODS` 目前只有：

```text
("legacy_rgb", "legacy + rgb")
```

VCP 跑完后需要加入：

```text
("rf_vcpa_rgb", "rf + VCPA + rgb")
```

如果补跑 `rf_rgb` seg baseline，也加入：

```text
("rf_rgb", "rf + rgb")
```

最终产出仍然写到：

```text
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/seg_compare_long.csv
analysis_outputs/20260626_dataset_update_rerun/01_results_tables/seg_compare_summary.csv
analysis_outputs/20260626_dataset_update_rerun/03_summaries/seg_findings.md
```

但文件内容必须明确标注：

```text
这些表对应的是“训练样本更新后”的 seg 重训结果。
```

### 9.6 已完成的 dense-only seg 实验

目前已切换为更小的验证口径：

```text
只启用 dense_mask_head
不启用 VCPA
不启用 visual_adapter / LoRA / score_fusion_head
```

目的：

- 单独验证“显式 mask 监督”能不能提升 `p_roc`。
- 不把 image-level 主线和 seg 支线搅在一起。

当前实验目录：

```text
analysis_outputs/20260626_dataset_update_rerun/runs_seg_refresh/legacy_rgb_densemask/
analysis_outputs/20260626_dataset_update_rerun/00_logs/seg_refresh_legacy_rgb_densemask/
analysis_outputs/20260626_dataset_update_rerun/03_summaries/seg_densemask_experiment.md
```

当前命令：

```bash
python run_rf_split_all_seg.py \
  --datasets burst_signal chirp_signal pulse_signal \
  --gpus 0 1 2 3 \
  --epochs 50 \
  --seed 111 \
  --root-dir analysis_outputs/20260626_dataset_update_rerun/runs_seg_refresh/legacy_rgb_densemask \
  --log-dir analysis_outputs/20260626_dataset_update_rerun/00_logs/seg_refresh_legacy_rgb_densemask \
  --prompt-mode legacy \
  --input-mode rgb \
  --batch-size 400 \
  --dense-mask-branch \
  --force
```

结果：

| method | burst_signal | chirp_signal | pulse_signal | overall |
|---|---:|---:|---:|---:|
| `legacy + rgb + dense-mask` | 95.8817 | 96.5267 | 91.5483 | 94.6522 |

Per-JSR：

```text
burst : m10=97.3250, m20=96.7475, m30=93.5725
chirp : m10=99.0600, m20=98.2900, m30=92.2300
pulse : m20=99.2175, m30=97.4900, m40=77.9375
```

当前判断：

- dense-only 不是通用增益。
- `chirp` 基本持平，但 `burst` 明显回退。
- `pulse m40` 很不稳定，是这条线最明显的短板。
- 当前不建议继续把 `dense_mask_head only` 当成主推进方向。

补充说明：

- `train_seg.py` 现已真正接入 `dense_mask_head` 的 BCE mask 监督。
- 开启 `dense_mask_head` 时，seg 评估改为直接使用 dense map 计算 `p_roc`。
- baseline 默认行为不变；只有显式传 `--dense-mask-branch` 才会进入这条线。
- 这轮结果对应新训练样本版本；如果要做严格对照，还需要在同一份新训练样本上补跑 `legacy_rgb` seg baseline。

### 9.7 暂不做的其它 VCP/dense 扩展

第一轮不要同时上这些：

```text
visual_adapter
visual_lora
text_aligned_dense
score_fusion_head
```

原因：

- 当前目标先缩小成：验证 `dense_mask_head` 本身对 Pixel-AUROC 是否有用。
- 同时上多个分支会导致无法判断到底是谁起作用。
- `dense_mask_head` 现在已经有可运行入口，但还需要看完整实验结果，不能提前当成已验证有效的成熟方案。
- `text_aligned_dense` 更适合之后单独做 APRIL-GAN 风格 map-only 对照，不要和 VCPA 第一轮混跑。

## 10. 明确禁止事项

不要做这些事：

- 不要跑 dsss。
- 不要跑 wideband。
- 不要跑跨库 public source -> target test。
- image-level 主线不要引入 VCP/dense branch；seg 支线按第 9 节计划单独做。
- 不要启用 visual adapter。
- 不要启用 LoRA。
- 不要启用 score fusion head。
- 不要把旧数据结果和新数据结果混在一个表里。
- 不要根据单个 scene 或单个 JSR 下结论，必须看三类总平均和每类平均。

## 11. 出错处理

如果 OOM：

1. 先把 `--batch-size 400` 降到 `200`。
2. 仍然 OOM 再降到 `100`。
3. 记录降 batch 的方法名、dataset、scene、JSR。
4. 不要改数据协议来规避 OOM。

如果某个 job 失败：

1. 保留失败日志。
2. 单独重跑这个 job 或该方法。
3. 最终 summary 里必须说明是否存在缺失项。

## 12. 下一步建议

当前不要再优先做手工输入变换。下一步分两条：

```text
Image 主线：
1. 使用 rf + rgb。
2. 补 pulse 的 RF prompt 识别，让 rf prompt 对 pulse 真正生效。
3. 保持 input_mode=rgb，不引入 morph/edge/local residual。

Seg 支线：
1. 先用新训练样本整套重训 legacy_rgb seg baseline。
2. 给 train_seg.py / run_rf_split_all_seg.py 接 VCPA 参数。
3. 基于同一份新训练样本跑 rf_vcpa_rgb，比较 Pixel-AUROC。
```
