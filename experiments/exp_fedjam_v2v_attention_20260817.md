# FedJam：V2V Attention 最终特征探索（2026-08-17）

## 目的

验证 PromptAD 中 V2V（Visual-to-Visual）attention 产生的最终 patch 特征，是否比当前 ViT-local 的中间层特征更适合频谱异常检测。

## 特征与协议

- 特征：`PromptAD` 的 `V2VTransformer` 输出 `visual_features[1]`，即最终 V-V 注意力路径的 patch tokens；V-V 路径中使用 `q=k=v`。
- 骨干：PromptAD `ViT-B-16-plus-240`，使用现有 checkpoint；不训练参数。
- 支持集：训练集中的正常样本，seed=111，嵌套 1/2/4-shot。
- 评分：正常 patch memory、1-NN cosine distance、patch 分数取最大值。
- 测试集：FedJam 官方独立测试集，共 7200 张（1800 正常、5400 异常）。

## 结果

| shot | AUROC (%) | AUPRC (%) | FPR@95%TPR (%) |
|---:|---:|---:|---:|
| 1 | 79.38 | 92.54 | 90.28 |
| 2 | 80.68 | 92.87 | 84.56 |
| 4 | 81.06 | 92.95 | 84.06 |

## 与现有 ViT-local 对比

| shot | 当前 ViT-local AUROC (%) | V2V-final AUROC (%) | 差值（V2V−ViT-local） |
|---:|---:|---:|---:|
| 1 | 84.42 | 79.38 | -5.04 |
| 2 | 84.76 | 80.68 | -4.08 |
| 4 | 85.33 | 81.06 | -4.27 |

## 结论

V2V 最终 patch 特征可以正常提取并完成少样本检测，但在 FedJam 上三种 shot 都低于当前 ViT-local，且 FPR@95%TPR 更高。因此暂不将 V2V 作为主线创新或主表方法，只保留为探索性特征分支结果。V2V attention 本身来自 PromptAD，这次实验的意义是验证其最终特征是否适合我们的频谱协议，而不是声称该注意力机制由本项目提出。

## 结果文件

- [评估脚本](../tools/eval_fedjam_v2v_patch_branch.py)
- [完整结果目录](../autoresearch/v2v-fedjam-260817/)
- [指标表](../autoresearch/v2v-fedjam-260817/metrics.csv)
- [汇总结果](../autoresearch/v2v-fedjam-260817/summary.json)
