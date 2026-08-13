# Safe support-only formalization log

日期：2026-07-31

## 假设

将置信度融合的参考分布从测试批次移到正常 support 后，仍可以保留相对 PromptAD 的主要
检测优势，同时消除 test-batch transductive 泄漏疑虑。

## 变更

- `utils/confidence_gate.py`：加入冻结的 support-only confidence rule（support CNN rank
  `q=0.8`、temperature `2.5`、alpha `2.5`）。
- `tools/eval_cls_dual_visual_evidence_fusion.py`：默认使用 support-only；旧
  transductive 规则只能通过兼容参数 `--gate-protocol transductive` 显式调用。
- `tools/eval_cls_ofdma_target_scene_ours.py`：默认使用 support-only；支持读取或在
  当前运行中生成正常 support 参考视图。
- `tools/eval_support_only_rf_gate.py` 与 `tools/eval_ofdma_support_only_gate.py`：
  生成正式 safe score bundle，同时保留 naive/transductive 对照。
- `tools/plot_paper_experiment_figures.py`：改用 safe RF/OFDMA 汇总结果；当前 OFDMA
  正式 ViT memory 使用 autoresearch 通过验证后冻结的 `top-k=1`。

## 复现命令

```bash
python tools/eval_support_only_rf_gate.py \
  --test-score-root analysis_outputs/20260731_rf_dual_visual_scores/scores \
  --vit-reference-root analysis_outputs/20260801_rf_support_only_vit_combined \
  --cnn-reference-root analysis_outputs/20260801_rf_support_only_cnn_combined \
  --output-root analysis_outputs/20260802_rf_support_only_formal

python tools/summarize_rf_five_type.py \
  --output-root analysis_outputs/20260802_rf_support_only_summary/summary

python tools/eval_cls_ofdma_target_scene_ours.py \
  --dataset-root /mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic \
  --split test --shots 1 2 4 \
  --output-root analysis_outputs/20260731_ofdma_target_scene_ours_topk1_formal \
  --gpu-id 1 --gate-protocol support_only

python autoresearch/replace_ofdma_ours_bundle.py \
  --baseline-root analysis_outputs/20260802_ofdma_v2_realistic_unified_safe \
  --ours-root analysis_outputs/20260731_ofdma_target_scene_ours_topk1_formal \
  --output-root analysis_outputs/20260731_ofdma_v2_realistic_unified_topk1_verified

python tools/plot_paper_experiment_figures.py \
  --output-dir analysis_outputs/20260731_paper_experiment_figures_topk1
```

## 结果

| 数据集 | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|
| In-house RF（本项目自测数据 + 合成干扰） | 91.967 | 81.801 | 22.000 |
| Public RF [14](https://doi.org/10.1007/s11036-009-0199-9) | 80.096 | 44.586 | 49.645 |
| OFDMA [15](https://arxiv.org/abs/2606.02102) 1-shot | 85.295 | 88.801 | 69.767 |
| OFDMA [15](https://arxiv.org/abs/2606.02102) 2-shot | 89.533 | 92.008 | 57.633 |
| OFDMA [15](https://arxiv.org/abs/2606.02102) 4-shot | 92.753 | 94.546 | 45.900 |

相对 PromptAD，RF 两个数据集的 AUROC 分别提升 1.44 和 1.02 个百分点；OFDMA
2/4-shot 的 AUROC 分别提升 0.66 和 1.24 个百分点。OFDMA 1-shot 略低于 PromptAD，
因此不能宣称 confidence fusion 在所有 shot 下都最优。

## 结论

结果支持将 support-only confidence fusion 作为论文正式协议。OFDMA 的 top-k=1 经过 validation 选择并
在完整 30 场景 4-shot test 上独立复核后固定。transductive 结果仍可用于说明批次
统计的上界，但不再进入主方法表述。多 support seed 的均值、标准差和固定源域校准仍
待补充，当前结果不声称已经完成这些统计检验。

数据集参考文献：

13. [FedJam 原论文](https://arxiv.org/abs/2508.09369)、[数据集](https://huggingface.co/datasets/panitsasi/FedJam)、[代码](https://github.com/panitsasi/fedJam)。
14. [Wellens–Mähönen 论文](https://doi.org/10.1007/s11036-009-0199-9)、[公开测量数据入口](http://download.mobnets.rwth-aachen.de)。
15. [OFDMA 论文](https://arxiv.org/abs/2606.02102)、[源码](https://github.com/akdd11/ofdma-spectrum-anomalies-simulation)、[Zenodo 数据](https://doi.org/10.5281/zenodo.20341906)。
