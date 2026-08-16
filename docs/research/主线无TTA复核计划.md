# 历史计划：主线无 TTA 复核

> 状态：已取消。当前决定是保留 TTA 作为辅助正常记忆扩充，不将其作为核心创新；现行计划见
> [`TTA 定位与复核计划`](TTA定位与复核计划.md)。本文件仅保留此前决策过程。

## 一、结论：需要重跑，但只重跑必要部分

需要重跑本文方法 Ours 的正式结果。原因是当前 In-house RF、Public RF 和 OFDMA 的历史主表
产物仍带有旧 Paired TTA，而现在确定的主线不使用 TTA。不能把旧数值直接改名为“无 TTA”。

不需要重跑传统通信基线、PatchCore、WinCLIP、VAE、SAIFE、UDMA 等外部对照：它们不依赖
本文的 TTA 配置。也不重新搜索 ViT 层、CNN 层、门控参数、PCA 或新融合公式。

## 二、冻结的正式方法

- 目标域免训练：不更新 ViT、CNN 或任何融合参数；
- 输入：只编码原始 support 与原始 query，`TTA=none`；
- ViT：`layer1 + layer2` patch 特征，50% farthest-first coreset；
- CNN：冻结 ImageNet ResNet18 `layer3`，完整局部 memory；
- 融合：固定 support-only confidence fusion，参数 `q=0.8`、`T=2.5`、`alpha=2.5`；
- 数据划分：复用已保存的 support manifest、测试文件列表、shot 设置和随机种子；
- 评价：AUROC、AUPRC、FPR@95%TPR；不使用测试批次统计量。

任何一项改变都必须单独记录为新实验，不得混入这次复核。

## 三、执行顺序

1. **代码与小样本校验**：确认四个 Ours 入口的正式 memory/query 均写入 `TTA=none`，并验证
   support/test 文件哈希与历史协议一致。FedJam/OFDMA 旧实现中用于置信度校准的派生 support
   reference view 必须替换为 leave-one-support-out 的原图评分，或单独标注为校准参考变换；在
   此前不得把产物称为完全无 TTA。
2. **In-house RF**：复跑 60 个 signal/scene/strength 单元，先产出 ViT-only、CNN-only、
   confidence fusion 的统一分数；这是主表和核心消融的优先项。
3. **Public RF**：在相同 `k=1/2/4-per-frequency` manifest 上复跑 Ours，更新主表和 shot 曲线。
4. **OFDMA**：在原 30 个 target scene 和 1/2/4-shot 协议上复跑 Ours；保留 21 SU 最大聚合，
   仅移除 ViT Paired TTA。
5. **FedJam**：以完整测试集复跑 1/2/4-shot Ours，使用 identity-only ViT memory；不再把 TTA
   reference views 混入正式校准。
6. **汇总与替换**：仅在四组结果齐全且配置核对通过后，统一更新论文表、图、README 与结果索引。

## 四、通过条件与止损

- 每个输出必须保存 `TTA=none`、manifest 路径、测试样本数和测试文件哈希；
- 所有数据集使用同一冻结融合规则；
- 若无 TTA 使某个数据集明显下降，照实报告，不为该数据集重新打开 TTA；
- 若双分支融合在某数据集不优于 ViT-only，保留为内部消融结论，不宣称普遍增益；
- 重跑结束前，不再启动任何新的结构、TTA、PCA、虚警控制或特征扩充探索。

## 五、重跑完成后的下一步

结果稳定后，工作重点转为：核对文献缺口、精炼问题设定、完善消融与统计稳定性，而不是继续
添加模块。只有在文献复核发现一个明确且可检验的频谱冷启动缺口时，才启动下一项技术创新探索。
