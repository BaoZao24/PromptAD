# 频谱时间 + 背景噪声底 Paired-TTA 探索记录（2026-08-13）

## 目的

验证一个专门面向频谱图的四视图 Paired-TTA：原图、时间方向双向平移、低功率背景噪声底
抖动。设计假设是：STFT 窗口起点/采集对齐会造成正常时间位置变化，接收机状态和环境会
造成正常噪声底变化，而高功率信号结构应保持不变。

## 实现

- RF bundle：`rf_spectral_time_background_v1`
  - `identity`
  - `rf_time_shift_up`
  - `rf_time_shift_down`
  - `rf_background_noise_jitter`
- OFDMA bundle：`ofdma_spectral_time_background_v1`
  - `identity`
  - `ofdma_time_shift_left`
  - `ofdma_time_shift_right`
  - `ofdma_background_noise_jitter`
- 时间平移为 4 px。
- 背景区域使用每张图灰度值 65% 分位数估计；噪声底扰动默认强度为 3 个灰度级，使用
  确定性的平滑、零均值局部场，只作用于低功率背景。
- support 和 query 使用同一组变换；融合为逐视图 max；CNN layer3 分支不加入新 TTA。
- 现有正式默认 bundle 没有修改。

## 预览与代码检查

- 预览脚本：`tools/preview_spectral_time_background_tta.py`
- 预览目录：`analysis_outputs/20260813_spectral_time_background_tta_preview/`
- 通过 `python -m py_compile`：`utils/spectral_tta.py`、RF/Public RF/OFDMA 评估入口及预览脚本。
- 合成测试确认输出尺寸和 dtype 不变，亮信号测试区域不被背景抖动直接修改。

## 受控结果

所有运行使用单 GPU、`num-workers=0`；RF 和 Public RF 使用固定 support manifest，OFDMA
使用同一场景、同一 shot、同一测试观测子集。

| 数据/设置 | 旧版 TTA | 新四视图 | 新版配置 |
|---|---:|---:|---|
| In-house RF burst，3 cell，ViT patch max AUROC | 99.48 | 95.41 | 背景强度 3 |
| In-house RF burst，3 cell，ViT patch top-0.05 AUROC | 97.53 | 94.75 | 背景强度 3 |
| Public RF burst，3 cell，ViT patch max AUROC | 87.61 | 87.65 | 背景强度 3 |
| Public RF burst，3 cell，ViT patch top-0.05 AUROC | 81.77 | 81.89 | 背景强度 3 |
| OFDMA `test_000`，1-shot，小规模 smoke，overall AUROC | 100.00 | 95.00 | background `support_only` |

RF 背景强度扫描（同一 smoke 设置）的 max AUROC：

| 强度 | 2 | 3 | 4 |
|---:|---:|---:|---:|
| AUROC (%) | 94.89 | 95.41 | 95.38 |

## 运行目录

- RF：`analysis_outputs/20260813_spectral_time_background_tta_experiment/rf_smoke_noise2/`
- RF：`analysis_outputs/20260813_spectral_time_background_tta_experiment/rf_smoke_noise3/`
- RF：`analysis_outputs/20260813_spectral_time_background_tta_experiment/rf_smoke_noise4/`
- Public RF：`analysis_outputs/20260813_spectral_time_background_tta_experiment/public_rf_burst/`
- OFDMA 旧版：`analysis_outputs/20260813_spectral_time_background_tta_experiment/ofdma_test000_old/`
- OFDMA 新版：`analysis_outputs/20260813_spectral_time_background_tta_experiment/ofdma_test000_time_background/`

## 结论

该 TTA 已经实现并通过了跨 RF、Public RF、OFDMA 的受控运行，且设计理由与频谱坐标、噪声
底变化相对应。但当前结果不是所有数据集都提升：RF smoke 和 OFDMA 单场景 smoke 下降，
Public RF burst 只有小幅提升。因此目前将其定位为“有针对性的探索设计”，不替换正式主线，
不在论文中宣称已经带来普遍性能增益。后续若要升级，应在独立 validation scene 上确定
视图和强度，再锁定 test 评估。
