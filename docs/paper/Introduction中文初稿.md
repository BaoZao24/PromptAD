# Introduction（中文初稿）

无线频谱承载着移动通信、卫星导航、雷达感知和物联网等多类无线业务。随着频谱接入设备和业务类型持续增加，非法发射、非预期干扰、设备故障和频谱误用更容易破坏合法信号的正常时频结构，进而降低通信质量并威胁频谱使用安全[1]–[4]。频谱异常检测通过持续分析功率谱或时频观测，识别偏离正常通信状态的异常模式，已成为自动化频谱监测的重要环节。

实际部署中的关键困难并非只有异常类型复杂，还包括目标场景数据的获取顺序。检测器迁移到新的站点、工作频段或设备后，通常能够先获得少量经确认正常的频谱图，而异常样本尚未出现；同时，噪声底、接收功率、合法业务占用和传播条件均可能随场景变化。在这种条件下，依赖异常标注的监督学习难以直接应用，依赖大量目标场景数据训练的模型也难以快速建立稳定的正常分布。因此，本文研究**少样本正常冷启动频谱异常检测**：仅利用目标场景中的少量正常参考频谱图建立场景模型，并对后续频谱观测进行独立检测。

现有频谱异常检测方法主要包括统计检测和数据驱动建模两条路线。能量检测、恒虚警检测、信息量分析和统计显著性检验依据功率、背景分布或传播规律构造检测统计量[1], [5]–[7]，具有明确的信号处理含义，但其阈值和背景模型需要随噪声水平、频段占用及接收条件重新校准。数据驱动方法进一步利用正常频谱学习复杂模式：SAIFE通过对抗自编码器学习功率谱表示[2]，IAD-PER和噪声注意方法从频谱图重构误差中提取异常证据[8], [9]，时空预测方法利用历史序列的预测残差识别异常[10]，TFAM-AAE通过时频注意和双度量增强困难异常检测[11]。近期的UDMA、GRETEL和MARD分别引入知识蒸馏、记忆增强、图结构建模和多传感器信息，以提高异常敏感性及定位能力[3], [4], [12]。这些方法推动了正常频谱建模，但通常需要使用目标场景数据优化模型参数；当每个新场景只有少量正常参考时，重新训练难以兼顾部署效率与正常模式覆盖。

少样本正常冷启动还要求检测器同时处理两类互补变化：合法通信信号的整体时频布局可能发生结构性偏离，局部区域也可能出现短时、窄带或弱功率干扰。单一特征表示难以稳定覆盖这两种尺度，而直接合并全局与局部分数又容易放大局部分支对背景纹理的敏感性。为此，本文提出 **SpectraMemAD**，将目标场景适配从参数学习转向记忆学习。该方法冻结通用预训练编码器，仅使用少量正常参考频谱图建立两类场景记忆：视觉Transformer分支记录整体时频结构，卷积分支记录局部纹理模式；待测频谱图分别与两类正常记忆进行1-NN匹配，得到全局和局部异常分数；随后利用当前场景正常参考分布校准局部分支的参与条件，在保持整体判断稳定性的同时补充局部异常证据。整个目标场景部署过程无需更新网络参数。

本文的主要贡献如下：

1. **从参数学习转向记忆学习的免训练场景适配。** 本文面向新站点、新频段仅提供少量正常频谱图的冷启动条件，以正常特征记忆承载目标场景知识，通过记忆构建和近邻检索完成部署阶段适配。

2. **面向频谱图的全局—局部双尺度特征记忆。** 本文分别记录合法通信信号的整体时频布局和局部纹理模式，使结构性偏离与局部干扰在相应尺度上独立形成异常证据。

3. **由正常参考分布校准的分数融合。** 本文利用目标场景正常参考分布估计两类分数的相对可靠性，使局部分支在提供有效补充证据时参与最终判断，从而协调全局稳定性与局部敏感性。

## 本稿引用文献

1. M. Afgani, S. Sinanovic, and H. Haas, “The Information Theoretic Approach to Signal Anomaly Detection for Cognitive Radio,” *International Journal of Digital Multimedia Broadcasting*, 2010. [本地PDF](../../references/papers/spectrum_domain/17_afgani_2010_information_theoretic.pdf)
2. S. Rajendran, W. Meert, V. Lenders, and S. Pollin, “Unsupervised Wireless Spectrum Anomaly Detection With Interpretable Features,” *IEEE Transactions on Cognitive Communications and Networking*, 2019. [本地PDF](../../references/papers/spectrum_domain/22_saife_tccn_2019.pdf)
3. P. Qi, T. Jiang, J. Xu, J. He, S. Zheng, and Z. Li, “Unsupervised Spectrum Anomaly Detection With Distillation and Memory Enhanced Autoencoders,” *IEEE Internet of Things Journal*, 2024. [本地PDF](<../../references/UDMA/Qi 等 - 2024 - Unsupervised Spectrum Anomaly Detection With Distillation and Memory Enhanced Autoencoders.pdf>)
4. A. Hussain *et al.*, “GRETEL: A Graph Attention Network for Low-SNR Spectrum Anomaly Detection in IoT Communications,” *IEEE Internet of Things Journal*, 2026. [本地PDF](<../../references/Hussain 等 - 2026 - GRE℡ A Graph Attention Network for Low-SNR Spectrum Anomaly Detection in IoT Communications.pdf>)
5. H. Urkowitz, “Energy Detection of Unknown Deterministic Signals,” *Proceedings of the IEEE*, 1967. [DOI](https://doi.org/10.1109/PROC.1967.5573)
6. H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target Situations,” *IEEE Transactions on Aerospace and Electronic Systems*, 1983. [本地PDF](../../references/papers/05_rohling_1983_cfar.pdf)
7. S. Liu, Y. Chen, W. Trappe, and L. J. Greenstein, “ALDO: An Anomaly Detection Framework for Dynamic Spectrum Access Networks,” *IEEE INFOCOM*, 2009. [本地PDF](../../references/papers/spectrum_domain/16_aldo_2009.pdf)
8. Y. Tian *et al.*, “Unsupervised Spectrum Anomaly Detection Method for Unauthorized Bands,” *Space: Science & Technology*, 2022. [本地PDF](../../references/papers/06_ism_2022_vae_anomaly.pdf)
9. J. Xu, Y. Tian, S. Yuan, and N. Liu, “Noise Attention Based Spectrum Anomaly Detection Method for Unauthorized Bands,” 2021. [本地PDF](../../references/papers/spectrum_domain/19_noise_attention_2021.pdf)
10. C. Peng, W. Hu, and L. Wang, “Spectrum Anomaly Detection Based on Spatio-Temporal Network Prediction,” *Electronics*, 2022. [本地PDF](../../references/papers/spectrum_domain/20_spatiotemporal_prediction_2022.pdf)
11. H. Ji, T. Zhang, X. Qiao, H. Wu, and G. Gui, “TFAM-AAE-U$_k$: A Dual-Metric Spectrum Anomaly Detection Algorithm,” *IEEE Communications Letters*, 2024. [本地PDF](../../references/TFAM-AAE-Uk_A_Dual-Metric_Spectrum_Anomaly_Detection_Algorithm.pdf)
12. R. Zhang, T. Zhang, H. Wu, G. Ding, H. Ji, and J. Zhang, “Detection and Localization of Spectrum Anomaly with Multi-Sensors: A Memory-Augmented Reverse Distillation Method,” *IEEE Internet of Things Journal*, 2026. [本地PDF](../../references/Detection_and_Localization_of_Spectrum_Anomaly_with_Multi-Sensors_A_Memory-Augmented_Reverse_Distillation_Method.pdf)
