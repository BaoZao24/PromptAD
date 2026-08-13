# 频率响应 Paired-TTA 消融（2026-08-13）

## 目的

将当前通用的 Gaussian blur 视图替换为频谱专用的**频率响应校准视图**，并在不改变模型、正常样本、测试集、融合规则的前提下，判断每个组件是否有效。

频率响应视图模拟接收前端或天线带来的轻微频率选择性增益起伏：同一个频率 bin 在所有时间帧上同步地小幅抬高或压低，时间形状和异常几何结构不变。默认强度为 3 个灰度级 RMS，曲线零均值、固定随机种子。

## 固定消融组

| 组别 | bundle | 视图 |
|---|---|---|
| 原图 | `none` | 原图 |
| 时间对齐 | `rf_time_alignment_v1` / `ofdma_time_alignment_v1` | 原图 + 时间轴双向 `±4 px` |
| 频率响应 | `rf_frequency_response_only_v1` / `ofdma_frequency_response_only_v1` | 原图 + 频率响应校准 |
| 组合 | `rf_spectral_response_v1` / `ofdma_spectral_response_v1` | 原图 + 时间轴双向 `±4 px` + 频率响应校准 |

所有组使用 paired TTA：每个 query 只和施加同一变换的 normal memory 比较；逐视图异常图取最大值。CNN 分支、support-only 置信度融合和所有其他超参数保持冻结。

## 评估顺序

1. In-house RF：固定正式 support manifest，先以 3 个代表性 cell 做筛选。
2. Public RF：固定 `k=1` manifest，以完整固定测试池验证筛选结果。
3. OFDMA：仅在前两者不退化时，用 target-scene 1/2/4-shot 正式协议验证。
4. FedJam：作为补充验证，不参与候选选择。

## 接纳规则

- 不以单个 test cell 的最高数值选方案。
- 组合方案需在 RF 与 Public RF 的固定协议上均不低于当前 `stft_shift_blur`，才进入 OFDMA 正式验证。
- 若组合在任一数据集明显退化，保留当前正式 TTA；频率响应视图只作为已否决的探索记录。

## 当前状态

- 已实现：`utils/spectral_tta.py`。
- 已通过单元测试：形状、dtype、RF/OFDMA 轴向和每频率 bin 沿时间恒定的响应偏移。
- 已生成预览：`analysis_outputs/20260813_spectral_response_tta_preview/`。
- 尚未启动 GPU 评估：2026-08-13 当前所有 GPU 均有其他任务运行，避免抢占服务器资源。
