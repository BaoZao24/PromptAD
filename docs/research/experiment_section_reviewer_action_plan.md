# 论文实验部分审稿问题与整改计划

更新时间：2026-07-23

## 1. 目标

本文档把当前实验部分可能受到的审稿质疑整理成可逐项完成的任务。
后续每次只处理一个问题，完成代码检查、小规模验证和正式实验后，再更新
对应状态和论文结果。

当前决定采用“目标场景 support”协议：

> 测试哪个目标场景，就只使用该目标场景中少量、确认正常的样本建立
> normal memory。

不再把来自单一源场景的 normal support 描述为“每个目标环境都有 target
normal support”。

## 2. 什么是 support 协议

support 是模型在测试前允许看到的正常参考样本。support 协议需要明确：

1. 正常样本来自哪里；
2. 使用多少张；
3. 如何选择；
4. 是否和测试数据来自相同原始记录或重叠时间窗口；
5. 不同方法是否使用完全相同的 support。

## 3. 已确定的新 support 协议

### 3.1 self RF

每个测试单元由 `signal + scene + JSR` 确定。

- normal 背景由 `scene` 决定；signal 和 JSR 描述加入该背景的异常；
- 同一 scene 下所有 signal/JSR normal 先合并，并按解码后的像素哈希去重；
- 每个频段选择 1 张 normal 图像；
- 一个 scene 只建立一套 ViT/CNN normal memory，该场景的所有 signal/JSR
  测试单元共用；
- support 内容及其不同路径副本全部从该场景各单元的 test normal 中排除；
- 各单元的 abnormal 图像保持不变；
- support 与 test 必须按原始 recording 或不重叠时间块隔离，不能只排除同名
  PNG；
- ViT、CNN 和融合使用同一个带哈希校验的 support manifest。

这对应“目标场景冷启动”：例如测试博物馆时，只能看博物馆的少量正常图。
self RF 共建立 4 套 scene memory，而不是 1 套 pooled memory 或 60 套 cell
memory。

### 3.2 public RF

public RF 的 normal 数据按 `MeasRes_*` 测量记录组织，而异常测试单元没有与
某个 `MeasRes_*` 一一对应的场景标签。因此采用测量记录级隔离：

- 选定一个或少量完整 `MeasRes_*` 记录作为 target normal support 来源；
- 每个频段选择 1 张 normal 图像；
- test normal 必须来自未进入 support 的其他完整测量记录；
- abnormal test 保持当前固定划分；
- 使用多个 support record seed 重复实验；
- 不根据异常类型或 JSR 重新挑选 support。

在正式实现前，需要再次核对 public RF 数据来源和异常生成方式，确认这种
record-level 定义与数据语义一致。

### 3.3 OFDMA

OFDMA 使用新的 target-scene cold-start 主协议：

- 每个 target scene 是一个固定的通信配置和物理上下文；
- 每个 observation 包含同一时刻 21 个 SU 的同步频谱图；
- 1/2/4-shot 分别使用该 target scene 内前 1/2/4 个正常 support observation；
- support 只含正常样本，测试使用同一 target scene 内独立的 normal-test 和
  anomaly-test observation；
- validation split 包含 5 个 target scene，用于方法设计确认；
- test split 包含 30 个完全独立的 target scene，只在方案固定后计算最终结果；
- 每个 test scene 固定包含 100 个正常 observation，以及 barrage、deceptive、
  pilot、sweep、random-hop 各 20 个异常 observation；
- 主指标是每个 target scene 独立计算后再做 scene-macro 的 observation AUROC、
  AUPRC 和 FPR@95%TPR；
- 不做 full-shot。

## 4. 问题清单与完成标准

| 顺序 | 问题 | 简单解释 | 需要完成的工作 | 完成标准 | 状态 |
|---:|---|---|---|---|---|
| P0-1 | support 协议不一致 | 旧代码从体育馆选图后测试所有场景，而且不同目录中的相同正常图可能同时进入 support/test | 实现 target-scene support；按像素内容去重；生成统一 manifest；删除正式入口的旧 pooled 模式 | self RF 只建立 4 套 scene memory；support/test 像素哈希无重叠；ViT/CNN/融合 manifest 哈希一致 | 代码、smoke 和 seed=111 正式 self RF 重跑均已完成；Ours AUROC=95.06 |
| P0-2 | support/test 可能相关 | 相邻 PNG 可能来自同一原始记录或重叠时间窗口 | 按 recording/session/time block 检查和划分；统计重叠 | support/test 不共享原始记录或重叠窗口；检查脚本输出 0 个冲突 | 时间块隔离已完成；recording 级受原始数据限制 |
| P0-3 | 缺少独立验证集 | 如果根据 test AUROC 选择 layer、top-k、fusion 参数，相当于看考试答案选参数 | 建立 development/validation split；冻结参数后只运行一次 test | 配置文件记录参数来源；正式 test 不参与选参和 checkpoint 选择 | 已完成：OFDMA 使用5个 validation 场景确认第三高patch与21-SU最大聚合，30个 test 场景冻结后完成一次正式评估 |
| P0-4 | 置信度融合依赖整批测试数据 | 单张图的分数会随同批其他图片变化 | 将 support-only confidence fusion 固化为正式方法；历史 transductive 实现仅保留在内部复现日志 | 正式代码只用正常 support 参考分布；论文主表不依赖测试批次统计 | support-only confidence fusion 已接入 RF/OFDMA 正式结果；批次大小/异常占比敏感性待补 |
| P0-5 | 单次 support 不稳定 | 当前主要结果只使用一组 support | 最低采用 RF 5 个独立 support seed、OFDMA 3 个全局 seed，固定测试集并统一评估 1/2/4-shot；public RF 固定 support 候选池之外的 test 记录 | 报告 mean、std、95% CI；保存每个 seed 的 support manifest；重复之间 test 路径哈希一致；OFDMA 以 scene 为 bootstrap 单位 | 当前实现采用更保守的 RF 20-seed 与 OFDMA 5-seed，均已完成；报告 mean/std/CI、配对检验和 scene bootstrap，见 `docs/research/support_stability_experiment.md` |
| P0-6 | 正式结果来源不唯一 | 不同 final 目录中可能出现不同数字 | 建立唯一 official results 目录和自动汇总入口 | 论文表格、CSV、README 来自同一组原始 score；不存在 92.26/93.01 等版本冲突 | self RF 新 support 的统一目录和汇总已完成；public RF 历史结果仍单独标注 |
| P1-1 | 双视觉置信度融合收益不稳定 | 与传统视觉/通信 baseline 的结论不一致，OFDMA 融合还出现饱和 | 做逐数据集、逐异常类型帮助/伤害分析；统计 confidence correction 开启率 | 同时对比传统视觉方法与有出处的 spectrogram-domain communication statistics；ViT normal-memory 只作内部消融 | seed=111 self RF 已按同一 support 完成全基线重跑；传统统计入口见 `tools/eval_traditional_spectral_baselines.py` |
| P1-2 | baseline 公平性不足 | 预训练 ViT/CNN 和从头训练方法的数据先验不同 | 统一 support、输入和指标；列出预训练数据；补官方 PatchCore和至少一个近期 few-shot 方法 | 论文包含 baseline 设置表；每种方法的预训练和适配方式可核对 | 待处理 |
| P1-3 | 数据集说明不足 | 审稿人不知道图片数量、采集方式和划分 | 补数据统计、设备、STFT 参数、异常来源、分辨率、许可证 | 一张完整数据集统计表和一张 split/support 表 | 待处理 |
| P1-4 | 缺少定位指标 | 方法产生 anomaly map，但目前主要报告图像级指标 | 使用已有 mask 计算 Pixel-AUROC、Pixel-AUPRC、AUPRO | 至少在有可靠 mask 的数据集上报告定位结果 | 待处理 |
| P1-5 | 缺少实际阈值实验 | AUROC 不说明部署时阈值怎么定 | 只用 support normal 设置阈值，报告 TPR、FPR、precision、recall | 阈值不使用 test label；报告至少 95%/99% normal 分位设置 | 待处理 |
| P2-1 | 缺少效率结果 | 双 backbone 和四视图 TTA 可能开销较大 | 测量建库时间、memory、显存、FPS、参数量 | 与 ViT only 和 CNN only 放在同一张效率表 | 待处理 |
| P2-2 | 缺少定性和失败案例 | 目前难以看出 CNN 什么时候真正帮助 | 输出 input、GT、ViT map、CNN map、confidence correction、final map | 每种主要异常有成功和失败案例；包含 OFDMA 融合下降案例 | 待处理 |
| P2-3 | TTA 语义需要证明 | 同时移动 support/query 后再取最大值是否真的解决时间错位尚不清楚 | 核对各数据集时间轴方向；比较单视图、各子视图、mean/max、相对位移 | 论文描述与代码轴方向一致；消融支持最终 TTA 设计 | 待处理 |
| P2-4 | 正式置信度融合不得使用标签校准 | 旧入口曾保留基于 test normal label 的 oracle 分支 | 正式入口只允许固定的无标签 support-derived confidence rule，标签只计算最终指标 | 正式入口无法使用 test labels 生成、选择或调节分数 | 已完成 |

## 5. 统一实验规则

后续整改遵守以下规则：

1. 不覆盖现有正式结果，所有新协议使用新输出目录；
2. 先做数据清单和 smoke test，再启动完整实验；
3. 每次只改变一个主要因素；
4. support、validation、test 三者明确隔离；
5. test label 只能在所有分数生成后计算指标；
6. 不做 full-shot；
7. 所有命令、seed、support 路径、配置和结果都写入实验 README；
8. 没有统计支持时，不使用“显著提升”或“稳定提升”表述。

## 6. 推荐处理顺序

```text
P0-1 目标场景 support
  -> P0-2 recording/time 防泄漏
  -> P0-3 validation/test 隔离
  -> P0-4 唯一置信度融合
  -> P0-5 多 seed
  -> P0-6 唯一正式结果
  -> P1/P2 补充实验
```

第一步只解决 `P0-1`：实现和验证新的 target-scene support 协议。完成之前不重跑
大规模主结果，也不修改论文中的最终性能数字。

## 7. P0-1 代码实施记录

日期：2026-07-23

假设：

> self RF 的 normal 背景属于 scene；同一 scene 的 signal/JSR 单元应共用
> content-deduplicated support，而不同 scene 不应共用 memory。

实现：

- `utils/rf_scene_support.py`：生成、校验和读取 target-scene manifest；
- `tools/build_rf_target_scene_support_manifest.py`：独立清单生成入口；
- `tools/eval_cls_vit_patchcore_gallery.py`：支持逐 scene 建 ViT memory；
- `tools/eval_cls_aux_cnn_gallery.py`：self RF 固定按 scene 建正式
  ResNet18 auxiliary CNN memory；
- `tools/eval_patchcore_cls.py`：独立 PatchCore baseline，同样遵守
  target-scene support，但不进入本文置信度融合；
- `tools/eval_cls_dual_visual_evidence_fusion.py`：拒绝融合不同 manifest 的分数；
- `run_sampling_ablation_scheduler.py`：当前实验固定使用 target-scene support，
  不再暴露旧 pooled 协议开关。

数据清单检查：

- 4 个 scene 各选择 24 张 per-frequency support；
- 共覆盖 60 个 signal/scene/JSR cell；
- 4 套 support 与各自 test normal 的解码像素哈希重叠均为 0；
- full-shot 在 target-scene 协议中被显式禁止。

模型级 smoke test：

- 数据：仅 `Gymnasium_spectrum`；
- 每个 cell 只取 1 张 test normal 和 1 张 abnormal；
- ViT、CNN 和融合均完成 15 个 cell 的打分；
- CNN 只拟合 1 次 scene memory；
- 三个阶段保存的 manifest SHA-256 完全一致。

复现命令：

```bash
SMOKE_ROOT=/tmp/promptad_target_scene_smoke

python tools/build_rf_target_scene_support_manifest.py \
  --output "$SMOKE_ROOT/support_manifest.json" \
  --normal-sampling per_frequency \
  --seed 111 \
  --scenes Gymnasium_spectrum

python tools/eval_cls_vit_patchcore_gallery.py \
  --output-root "$SMOKE_ROOT/self_vit" \
  --support-manifest "$SMOKE_ROOT/support_manifest.json" \
  --normal-sampling per_frequency \
  --scenes Gymnasium_spectrum \
  --max-test-normals 1 \
  --max-abnormals 1 \
  --coreset-ratio 0.01 \
  --paired-tta none

python tools/eval_cls_aux_cnn_gallery.py \
  --protocol rf_target \
  --support-manifest "$SMOKE_ROOT/support_manifest.json" \
  --output-root "$SMOKE_ROOT/self_aux_cnn" \
  --normal-sampling per_frequency \
  --num-workers 0

python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol rf_target \
  --vit-score-dir "$SMOKE_ROOT/self_vit/scores" \
  --cnn-score-dir "$SMOKE_ROOT/self_aux_cnn/scores" \
  --output-root "$SMOKE_ROOT/self_fusion"
```

结论：P0-1 的代码路径支持该假设。smoke 指标不作为论文结果。

## 8. P0-2 时间块隔离实施记录

日期：2026-07-23

实验名：`smoke_rf_target_time_isolation_2026_07_23`

问题：

原始切片使用长度 4000、步长 2000 的滑动窗。若任意选择 support，再把其余图片
当作 test，即使文件不同，也可能来自重叠的原始时间片。

修正：

- support 只从正常图 `[0, 4000)` 中选择；
- normal 和 abnormal test 只保留 `t_start >= 4000`；
- 与 support 重叠的 `[2000, 6000)` 整段不进入 test；
- manifest 升级为 version 2，同时固定 support、test normal 和 test abnormal；
- validator 同时检查时间区间、解码像素哈希和 60 个实验单元；
- ViT 与 CNN 默认口径分别改为 `target_scene` 和 `per_scene`；
- 旧逐信号 loader 改为真实目录的兼容包装，不再读取不存在的
  `/datasets/normal/{scene}`，数据缺失时不再静默返回空集。

单场景数据 smoke：

- 场景：`Gymnasium_spectrum`；
- support：24 张，全部来自 `t=0–4000`；
- 15 个 signal/JSR cell 均同时保留 normal 与 abnormal test；
- test 最小 `t_start=4000`；
- ViT、CNN 和置信度融合均跑通，manifest SHA-256 均为
  `cf9a92bd2a4ceaec51c64a973dc6785b136c27dbfb12ff999f04270e4d8b45dc`。

验证命令：

```bash
SMOKE_ROOT=/tmp/promptad_rf_v2_model_smoke.d1MIUO

python tools/build_rf_target_scene_support_manifest.py \
  --output "$SMOKE_ROOT/support_manifest.json" \
  --normal-sampling per_frequency --seed 111 \
  --scenes Gymnasium_spectrum

python tools/eval_cls_vit_patchcore_gallery.py \
  --output-root "$SMOKE_ROOT/self_vit" \
  --support-manifest "$SMOKE_ROOT/support_manifest.json" \
  --normal-sampling per_frequency \
  --scenes Gymnasium_spectrum \
  --max-test-normals 1 --max-abnormals 1 \
  --coreset-ratio 0.01 --paired-tta none --num-workers 0

python tools/eval_cls_aux_cnn_gallery.py \
  --protocol rf_target \
  --support-manifest "$SMOKE_ROOT/support_manifest.json" \
  --output-root "$SMOKE_ROOT/self_aux_cnn" \
  --normal-sampling per_frequency \
  --num-workers 0

python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol rf_target \
  --vit-score-dir "$SMOKE_ROOT/self_vit/scores" \
  --cnn-score-dir "$SMOKE_ROOT/self_aux_cnn/scores" \
  --output-root "$SMOKE_ROOT/self_fusion"
```

结论：P0-2 的数据与代码级检查通过；尚未重跑正式性能表，smoke 结果不作为论文
指标。该结论只表示时间区间不重叠；当前每个场景只有一段原始采集，support 与 test
仍来自同一 recording。若要求 recording-level 隔离，需要为每个场景补采独立的
正常 recording。完整协议见 `docs/research/rf_target_data_protocol.md`。
