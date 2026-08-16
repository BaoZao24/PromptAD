# tools/ 脚本索引

本目录按用途分组。废弃/一次性探索脚本已移入 `archive/`，不再出现在根目录。

## 正式协议（当前论文主线）

| 脚本 | 作用 |
|---|---|
| `build_rf_target_scene_support_manifest.py` | 生成 self-RF target-scene support manifest（正式协议入口） |
| `eval_cls_vit_patchcore_gallery.py` | ViT patch normal memory 主证据（farthest-first coreset + top-k） |
| `eval_cls_aux_cnn_gallery.py` | CNN 局部 memory 辅助证据（完整 memory + 单近邻） |
| `eval_cls_dual_visual_evidence_fusion.py` | 置信度融合（ViT 主 + CNN 门控补充），当前正式方法；不启用 TTA |

## 正式基线（论文对照）

| 脚本 | 基线 |
|---|---|
| `eval_patchcore_cls.py` | PatchCore（官方实现基线） |
| `eval_padim_cls.py` | PaDiM |
| `eval_stfpm_cls.py` | STFPM |
| `eval_deepsvdd_cls.py` | DeepSVDD |
| `eval_vae_cls.py` | VAE |
| `eval_winclip_main.py` | WinCLIP |
| `eval_traditional_spectral_baselines.py` | 传统频谱统计（ED/谱熵/谱平坦度/峰度/CA-CFAR） |
| `eval_information_theoretic_baselines.py` | 信息论基线 |
| `eval_udma_cls.py` | UDMA |
| `eval_fastrecon_cls.py` | FastRecon |
| `eval_saife_rf_cls.py` | SAIFE |

## Ours 各数据集变体（内部消融/对照）

| 脚本 | 作用 |
|---|---|
| `eval_cls_public_rf_vit_patchcore_gallery.py` | Public RF 版 ViT gallery |
| `eval_cls_spectrum_vit_nn_gallery.py` | Spectrum 版 ViT gallery |
| `eval_cls_vit_patch_gallery.py` | ViT patch gallery（无 coreset 简化版） |
| `eval_cls_dinov2_patchcore_gallery.py` / `eval_cls_public_rf_dinov2_patchcore_gallery.py` | DINOv2 backbone gallery |
| `eval_cls_vit_guided_cnn_gallery.py` | ViT 引导的 CNN gallery |
| `eval_cls_resnet_gallery_fusion.py` / `eval_seg_resnet_gallery_fusion.py` | ResNet gallery 融合 |
| `eval_cls_spectrum_dual_gallery.py` / `eval_cls_public_rf_dual_gallery.py` | 双 gallery |
| `eval_cls_public_rf_multibranch_harmonic.py` / `eval_cls_multibranch_harmonic_from_scores.py` | 多分支谐波融合 |
| `eval_rf_feature_fusion.py` | RF 特征融合 |
| `eval_support_only_rf_gate.py` / `eval_support_only_safe_gate.py` | support-only 门控 |
| `eval_seg_dual_gallery_fusion.py` / `eval_seg_prompt_normal_gallery_fusion.py` | seg 分支 fusion |

## OFDMA target-scene 家族

| 脚本 | 作用 |
|---|---|
| `generate_ofdma_target_scene_dataset.py` | 生成 OFDMA target-scene 数据集 |
| `launch_ofdma_target_scene_generation.py` | 数据集生成调度 |
| `validate_ofdma_target_scene_dataset.py` | 数据集校验 |
| `make_real_anomaly_maps.py` | 生成真实异常图 |
| `build_ofdma_support_manifest.py` / `build_public_rf_support_manifest.py` / `build_public_rf_k_per_frequency_manifests.py` | support manifest 生成 |
| `ofdma_fewshot_baseline_common.py` | few-shot baseline 公共代码 |
| `eval_cls_ofdma_fewshot_comparison.py` | OFDMA few-shot 对比 |
| `eval_cls_ofdma_target_scene_ours.py` | target-scene Ours 评估；默认 identity-only、无 TTA |
| `eval_ofdma_target_scene_baselines.py` | target-scene 基线评估 |
| `eval_ofdma_support_only_gate.py` | support-only 门控 |
| `eval_saife_ofdma_fewshot.py` | SAIFE few-shot |
| `merge_ofdma_method_chunks.py` / `merge_ofdma_v2_results.py` | 分块结果合并 |
| `summarize_ofdma_fewshot_baselines.py` | 结果汇总 |

## FedJam 家族

| 脚本 | 作用 |
|---|---|
| `eval_fedjam_fewshot_dual.py` | FedJam few-shot 双分支；默认 identity-only、无 TTA |
| `eval_fedjam_visual_baselines.py` | FedJam 视觉基线 |
| `eval_fedjam_traditional_spectral.py` | FedJam 传统频谱基线 |
| `eval_fedjam_four_branches.py` | FedJam 四个独立视觉分支：ViT-local、ViT-global、CNN-local、DINO-local；不做融合 |
| `monitor_gpu_and_run_fedjam_branches.py` | 安全轮询空闲 GPU，满足阈值后自动启动四分支实验 |

## 频谱 TTA（辅助记忆扩充，不属于核心创新）

| 脚本 | 作用 |
|---|---|
| `preview_spectral_background_tta.py` | 频谱背景 TTA 预览 |
| `preview_spectral_response_tta.py` | 频谱频率响应 TTA 预览 |
| `preview_spectral_structure_tta.py` | 频谱结构 TTA 预览 |
| `preview_spectral_time_background_tta.py` | 时频背景 TTA 预览 |

`utils/spectral_tta.py` 和带 TTA 参数的评估入口用于复现辅助记忆扩充实验。TTA 不写入架构图
或核心方法图；启用时必须与相同划分的无 TTA 对照一并报告。

## UniVAD 适配实验（探索性）

| 脚本 | 作用 |
|---|---|
| `eval_univad_rf_texture_b_adapted.py` | 使用本机 DINOv2-B 的 UniVAD 整体纹理路径适配；不含 C³/GECM，不是官方完整复现 |
| `eval_univad_rf_fewshot.py` | 尝试官方 DINOv2-G/UniVAD 入口；需要约 4.23 GB 权重，不作为默认实验入口 |
| `eval_univad_public_rf.py` | Public RF `k=1/2/4-per-frequency` 全量纹理适配；复用固定正常 test 池，避免重复编码 |
| `eval_univad_fedjam.py` | FedJam spectrogram-only benign 1/2/4-shot 全量 test 适配 |
| `eval_univad_ofdma.py` | OFDMA v2-realistic 21-SU observation 聚合适配；默认先跑一个完整 target scene |

## 汇总 / 图表 / 报告

| 脚本 | 作用 |
|---|---|
| `summarize_method_funnel.py` / `summarize_rf_five_type.py` / `summarize_public_rf_wideband.py` / `summarize_public_rf_k_per_frequency_metrics.py` / `summarize_sampling_shot_ablation.py` / `summarize_support_stability.py` | 各类结果汇总 |
| `plot_paper_experiment_figures.py` / `plot_support_stability.py` | 论文图表 |
| `render_experiment_pdf.py` | 实验报告 PDF 渲染 |
| `export_current_scheme_figures.py` / `export_final_visual_method_report.py` / `export_rf_dual_visual_scores.py` | 图表导出 |
| `run_support_stability_experiment.py` | support 稳定性实验调度 |

## archive/（已归档，废弃或一次性探索）

历史 fastrecon 互补性分析、6 月特征提取方案对比图导出、旧 winclip/dual_gallery/target_normal_gallery 评估、`run_original_promptad_rf_baseline.py` 等 23 个脚本。仅供追溯，不再使用。
