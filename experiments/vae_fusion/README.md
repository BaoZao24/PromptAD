# VAE Reconstruction Error 探索记录

日期: 2026-06-02
分支: exp/vae-fusion-experiment
状态: 已关闭（结论：不推荐作为第二创新点）

## 实验目标

调研 VAE reconstruction error 是否能作为 PromptAD 的辅助评分信号，
在 burst/chirp/dsss 三类异常上有稳定提升。

## 方法

- 使用已有 VAE 模型（卷积 VAE, 256×256, latent_dim=256）
- 在 RF_Spectrum_Public_Dataset 上预训练 1000 epochs
- 在 PromptAD 测试图片上计算逐图像 MSE reconstruction error
- 融合: final = PromptAD_score + beta * normalize(VAE_MSE)
- 对比基线: PromptAD no_contrast (text_only scoring)

## 数据

- 异常类型: burst_signal, chirp_signal, dsss_signal
- 场景: Gymnasium_spectrum, Playground_spectrum, TimeSquare_spectrum, WeaponMuseum_spectrum
- 噪声: m10db, m20db, m30db
- 协议: normal_75_25, seed=111

## 结果

### Burst (12 实验) — 全部失败
- VAE AUC: 0.38-0.55
- 融合后全部比 PromptAD 单独更差
- 关键: abnormal 的 MSE 低于 normal 的 MSE（方向反向）

### Chirp (9 实验) — 全部失败
- VAE AUC: 0.47-0.50
- 同样方向反向

### DSSS (12 实验) — 6/12 有微弱提升
- Best: Playground/m10db: 0.8503 → 0.9052 (+0.055)
- Overall DSSS: 0.760 → 0.772 (+0.012)
- 提升太小且不一致

## 失败原因分析

### 根本原因: MSE-based reconstruction error 是全局度量
VAE 在整张图上计算逐像素 MSE，然后求和。这会：

1. 让背景噪声主导总误差 —— 局部异常信号被"平均掉"
2. 对 burst/chirp: 异常 = 局部结构化信号 → VAE decoder 能较好重建 → MSE 反而更低
3. 对 DSSS: 异常 = 全局能量变化 → VAE 难以重建 → MSE 方向正确（但判别力有限）

### 和 global normal distribution scoring 失败的共性
两个方法都失败在同一个点上:
- global normal distribution: 用 global CLIP feature 做距离度量 → 丢失局部信息
- VAE reconstruction error: 用全局像素 MSE 做异常分数 → 丢失局部信息

这从反面验证了: patch-level score aggregation 才是正确方向。

### DSSS vs burst/chirp 不对称性
```
burst: 时域窄带脉冲 → VAE 认为 = 干净的竖线 → 容易重建 ✓ (反向)
chirp: 频率扫描斜线 → VAE 认为 = 干净的斜线 → 容易重建 ✓ (反向)
DSSS: 扩频噪声 → VAE 认为 = 脏信号 → 难以重建 ✗ (正向)
```

这说明 VAE 学到的是"图像结构完整性"而非"异常语义"。

## 建议

1. **不推荐 VAE reconstruction error 作为第二创新点**：对 burst/chirp 方向反向，DSSS 提升太小
2. **VAE 在 feature space 上可能有效**：不重建像素，而是在 CLIP 特征空间上做 normal distribution modeling
3. **建议转向 patch-level normal distribution**：复用 PromptAD 已有的 patch feature gallery，做 patch 级别的正常分布建模
4. **DSSS 的发现值得注意**：DSSS 是唯一 VAE 能检测的异常类型，因为它是全局能量变化。这可能意味着 DSSS 需要和 burst/chirp 不同的检测策略。

## 代码位置

- 实验脚本: tools/vae_fusion_experiment.py
- VAE 模型: /mnt/data/wangbei/anomaly_detection/vae_ism_ano/model.py
- 预训练权重: /mnt/data/wangbei/anomaly_detection/vae_ism_ano/result_rf_public_256/best.pth
