# In-house RF：1-NN、无TTA重评估与图2更新

## 设置

沿用`20260728_rf_five_type_formal_seed111/support_manifest.json`，四个场景、五类干扰，
共60个评估单元；未限制测试正常样本或异常样本数量。seed=111。
清单SHA256：`65f4a06db8b490588020088fb2a522b960dc65d964096a2cede7d62ee3c6d296`。

- ViT-B/16-plus-240，冻结LAION预训练视觉编码器，第1、2特征层，50%最远点子集，1-NN，最大距离评分。
- ResNet18，冻结ImageNet预训练权重，layer3，50%最远点子集，1-NN，最高10%距离均值。
- 两条分支均无TTA，无正常参考图增强。
- 校准使用原始正常特征，排除精确自匹配；融合常数q=0.8、T=2.5、alpha=2.5。
- GPU1运行ViT，GPU2运行CNN；batch=16，每进程2个加载worker。未使用GPU0。
- 未加载历史prompt训练checkpoint；仅使用预训练视觉编码器的特征进行评分。

## 结果

| 分支 | AUROC | AUPRC（平均精确率） | FPR@95%TPR |
|---|---:|---:|---:|
| ViT | 91.269600 | 81.406866 | 24.983458 |
| CNN | 90.217087 | 78.446447 | 34.151814 |
| 融合 | 91.466569 | 81.428634 | 24.588398 |

图、表统一使用融合结果91.47/81.43/24.59。融合相对较强单分支提升0.20个百分点。
相对论文PatchCore行，AUROC和AUPRC提升7.09和11.26个百分点，FPR95降低31.35个百分点
（基于表中保留两位小数的值）。

此前表格91.97/81.80/22.00和预览图92.25/81.81/22.64均不作为本次新图的来源。
历史记录保留；新图三幅面板的Ours全部来自本次融合预测。其他数据集结果未重新运行。

## 产物

- 执行入口：`tools/run_inhouse_1nn_no_tta_20260911.py`。
- 实验输出：`analysis_outputs/20260911_inhouse_1nn_no_tta/`，含分支分数、校准参考、融合分数、日志与清单信息。
- 绘图入口：`tools/plot_inhouse_metric_triptych.py`。
- 图：`analysis_outputs/20260911_inhouse_metric_triptych/inhouse_metrics_formal.png`及同名PDF、JSON。
- 与其他三联图采用相同7.15×2.55英寸画布、400 dpi，论文中等比例单栏放置。

ROC和PR曲线按单元等权平均；GRETEL按其原有三次重复共180条曲线平均。
FPR95先在每个单元取TPR首次达到0.95的位置，再求均值。
JSON中的average_precision与论文AUPRC口径对应；插值宏平均PR曲线的几何面积不要求等于该指标。
