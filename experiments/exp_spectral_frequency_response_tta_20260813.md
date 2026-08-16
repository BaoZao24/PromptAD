# 频谱频响校正 TTA：无 TTA 对照

目的：验证频谱专用 TTA 是否至少优于不做 TTA 的正常记忆库。固定使用合并记忆库：对正常 support 构造多个视图后合并、测试图仅以原图查询一次。

统一频谱 TTA 为四个正常视图：原图、时间正移、时间负移、平滑零均值频率响应漂移。频响漂移幅度固定为 `3`；它模拟接收机或天线对不同频段的轻微增益/衰减变化，不移动或模糊信号事件。

## 结果

| 数据集 | 无 TTA AUROC | 频谱 TTA AUROC | 变化 | 结论 |
|---|---:|---:|---:|---|
| In-house RF（60 条件宏平均） | 92.3211 | 92.3214 | +0.0002 pp | 基本持平 |
| Public RF，k=1（15 条件宏平均） | 79.9349 | 80.2847 | +0.3498 pp | 提升 |
| Public RF，k=2（15 条件宏平均） | 80.8761 | 80.8505 | -0.0257 pp | 基本持平 |
| Public RF，k=4（15 条件宏平均） | 81.8158 | 81.7391 | -0.0767 pp | 基本持平 |
| FedJam，1-shot（融合） | 84.2761 | 85.1427 | +0.8667 pp | 提升 |
| FedJam，2-shot（融合） | 84.9933 | 85.8959 | +0.9026 pp | 提升 |
| FedJam，4-shot（融合） | 85.4036 | 86.1065 | +0.7028 pp | 提升 |
| OFDMA，1-shot（融合） | 87.2303 | 86.8713 | -0.3590 pp | 下降 |
| OFDMA，2-shot（融合） | 91.5530 | 91.2680 | -0.2850 pp | 下降 |
| OFDMA，4-shot（融合） | 94.4047 | 94.2617 | -0.1430 pp | 下降 |

结论：该设计在 FedJam 的各 shot 设置均优于无 TTA；Public RF 仅在最稀缺的 k=1 时提升，k=2、4 基本持平；In-house RF 基本持平；OFDMA 的 1/2/4-shot 则轻微下降。因此它不能表述为“所有数据集必然提升”，而应强调其在极少正常样本时扩充正常记忆覆盖范围的作用。

注意：本表验证的是完整四视图 TTA 包相对于无 TTA 的作用，不能把全部提升单独归因于频率响应漂移，因为该包同时含有两个时间偏移视图。若论文需要单独宣称频响漂移的增益，还应补做“仅原图 + 时间偏移”的成分消融。

## 可复现结果

- In-house RF：`analysis_outputs/exploratory/20260813_frequency_response_memory_rf_full/`
- Public RF k=1：`analysis_outputs/exploratory/20260813_frequency_response_memory_public_rf_k1/`
- Public RF k=2：`analysis_outputs/exploratory/20260814_frequency_response_memory_public_rf_k2/`
- Public RF k=4：`analysis_outputs/exploratory/20260814_frequency_response_memory_public_rf_k4/`
- FedJam：`analysis_outputs/exploratory/20260813_frequency_response_memory_fedjam_full/`
- OFDMA：`analysis_outputs/exploratory/20260813_ofdma_spectral_response_tta/`

所有对照均使用相同 support 清单、相同独立测试池、同一特征提取与评分规则。无 TTA 仅保留原图 normal support；频谱 TTA 增加三个正常视图，因此记忆库大小相应增加四倍。这正是 TTA 扩充正常记忆覆盖范围的实际机制。
