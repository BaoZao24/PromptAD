# Plan: 用 FastRecon 跑 pooled 4 类信号 few-shot cls 任务

## 目标
在项目 pooled 4 类信号 cls 协议上跑 FastRecon（few-shot，每频段一张 normal），输出 image-AUROC，
可与 PromptAD current rescore (78.4088) 在同一协议下对比。FastRecon 是 PatchCore 同骨架的重建类方法，
直接对标已落地的 `tools/eval_patchcore_cls.py`。

## FastRecon 方法（来自 references/FastRecon/main.py）
- 骨架 wide_resnet50_2（ImageNet 预训练，本地已缓存），hook layer2[-1]+layer3[-1]，avgpool(3,1,1) 后
  embedding_concat → patch 特征 [B, C=1536, H=14, W=14]（input 224）。
- Support：所有 normal 训练样本的 patch 特征汇成池 → kCenterGreedy 选 coreset `Sc` [Ns, C]；
  `mu` = 所有 support 图特征图的空间均值 [H*W, C]（= 正常分布的中心）。
- Query `Q` [H*W, C]：带分布正则的闭式回归
  `W = (Q·Scᵀ + λ·μ·Scᵀ) · inv((1+λ)·Sc·Scᵀ)`，重建 `Q̂ = W·Sc`，
  image score = `max_patch ‖Q − Q̂‖₂`。
- 目标等价于 `min ‖Q−W·Sc‖² + λ‖W·Sc−μ‖²`（重建既要像 query、又要靠近正常均值）。

## 与 FastRecon 源码的偏差（修正 bug）
源码 `training_step` 里 `self.embedding_list` / `self.embedding_list_mu_cor` 每个 batch 被覆盖，
batch_size=1 时 coreset 和 mu 只用了**最后一张** support 图。本实现聚合**所有** support 样本
（符合论文意图），否则 few-shot 下 mu 退化为单张图、重建正则失效。

## 协议（与 current rescore / patchcore 脚本完全一致）
- 信号: burst / chirp / dsss / pulse（pooled 训练，逐 (signal,scene,jsr) 评估）
- 场景: WeaponMuseum / Playground / TimeSquare / Gymnasium（4 个 _spectrum）
- JSR（rf_target，来自 `JSR_BY_SIGNAL`）: burst/chirp/dsss → m10/m20/m30；pulse → m20/m30/m40
- 切分: normal_75_25, normal_train_ratio=0.75, seed=111
- Few-shot normal 采样: `frequency_one_per_band`（每个频段一张正常样本）——用户指定
- 骨架权重 `wide_resnet50_2-95faca4d.pth` 已在 torch hub 缓存

## 实现步骤

### 1. 新建 `tools/eval_fastrecon_cls.py`
结构对标 `tools/eval_patchcore_cls.py`（同 CSV/summary/README 输出格式）。

**复用（从 `tools.eval_patchcore_cls` import，零拷贝）**：
- `PatchCorePathDataset`（resize 256 → CenterCrop 224 → ImageNet norm，PIL RGB 加载）
- `make_sample, safe_auc, write_csv, append_average, train_signature`
- `rf_target_jobs, public_rf_jobs, spectrum_jobs`（job 构建器，全部尊重 `--normal-sampling frequency_one_per_band`）

**新写（FastRecon 核心）**：
- `embedding_concat` / `avgpool` helper —— 原样搬自 FastRecon main.py。
- `FastReconEncoder`：torchvision `wide_resnet50_2(weights=IMAGENET1K_V1)`，`requires_grad_(False).eval()`，
  register_forward_hook 于 `layer2[-1]`、`layer3[-1]`；`forward(x)` 返回 concat 后的 [B,1536,14,14]。
- `fit_fastrecon(train_samples, args, device)`：
  1. DataLoader 遍历 support → encoder 提特征 → avgpool+concat；
  2. 累积所有 patch `[N,1536]` 与每图 feature map `[N_img,1536,196]`；
  3. `kCenterGreedy`（from `references/FastRecon/sampling_methods/kcenter_greedy.py`，加 sys.path）选 coreset → `Sc`；
  4. `mu = mean over N_img → [196,1536]`（转置后）；
  5. 预计算 `inv_temp = inv((1+λ)·Sc·Scᵀ)` [Ns,Ns] 与 `mu_ScT = mu·Scᵀ` [196,Ns]，存到 model 对象。
- `predict_job(model, job, args)`：
  1. 遍历 test DataLoader（batch），提 Q [B,196,1536]；
  2. `W = (Q·Scᵀ + λ·mu_ScT) @ inv_temp`，`Q̂ = W·Sc`；
  3. `score = max over 196 of ‖Q−Q̂‖₂` → [B]；
  4. 存 scores.npz，返回 row（dataset/category/scene/jsr/num_*/image_auroc）。
  - query 分块（chunk_size）控内存；inv_temp/mu_ScT 不依赖 query，只算一次。
- `run_jobs`：按 `train_signature` 分组（pooled → 全部 48 cell 共享一个 train set），
  每组 fit 一次、predict 该组所有 eval job（与 patchcore 脚本一致）。
- `main`：build jobs → run_jobs → append_average → 写 CSV + summary.json + README.md。
- **CLI 参数**（FastRecon 默认 + patchcore 对齐）：
  `--protocol`（默认 rf_target）、`--rf-train-mode`（默认 pooled）、`--normal-sampling`（默认 frequency_one_per_band）、
  `--lambda`（默认 2，FastRecon 论文值）、`--coreset-ratio`（默认 0.1）、
  `--resize 256 --imagesize 224 --backbone wideresnet50 --layers layer2 layer3`、
  `--batch-size 32 --num-workers 4 --seed 111 --normal-train-ratio 0.75 --gpu-id 0`、
  `--output-root analysis_outputs/20260703_fastrecon_cls`、`--max-test-normals 0 --max-abnormals 0`（全量）。

### 2. 冒烟测试
`python tools/eval_fastrecon_cls.py --protocol rf_target --rf-signals burst_signal --rf-scenes WeaponMuseum_spectrum --max-test-normals 20 --max-abnormals 20`
确认：权重加载、support 特征/coreset/mu 构建、回归重建、i_roc 计算全链路通、数值合理（非 NaN、score 有区分度）。

### 3. 全量跑
`python tools/eval_fastrecon_cls.py`（pooled 4 类，48 个评估组合），打印 per-signal 均值 + OVERALL 总均值 + vs 78.4088 的 delta。

## 输出
- `analysis_outputs/20260703_fastrecon_cls/results_fastrecon_cls.csv`：每行 method/dataset/category/scene/jsr/num_train_normal/num_test_normal/num_test_abnormal/image_auroc + 末行 macro 平均
- `summary.json`：方法/协议/骨架/λ/coreset_ratio/seed/image_auroc_macro
- `README.md`：说明来源 `references/FastRecon`、协议、与 current rescore 对比
- `scores/*.npz`：每个 eval cell 的 scores+labels+paths，供后续融合/分析

## 关键技术点
- 同骨架同 layer（wide_resnet50_2 / layer2+layer3）→ 与 `eval_patchcore_cls.py` 特征空间一致，结果可直接横向对比。
- 闭式解可批量：`inv_temp`[Ns,Ns]、`mu_ScT`[196,Ns] 预计算；query 批处理只做矩阵乘，无 per-sample 求逆。
- coreset 用 FastRecon 自带 kCenterGreedy（SparseRandomProjection 降维后贪心），忠实源码。
- 不修改 FastRecon 源码，保持 references 完整；kCenterGreedy 通过 sys.path 导入。
