# SpectraMemAD: Dual-Scale Normal Memory for Training-Free Few-Shot Spectrum Anomaly Detection

## Abstract

Spectrum anomaly detection supports interference monitoring, but deployment at a new site or frequency band often begins with only a few normal spectrograms. This scarcity limits target-scene model training, while detection must account for both global occupancy changes and local interference. We propose SpectraMemAD, which shifts scene adaptation from parameter learning to memory learning using frozen pretrained visual encoders. Vision Transformer (ViT) and convolutional neural network (CNN) features form separate normal memories for overall time-frequency structure and local texture. Nearest-neighbor distances produce spatial anomaly maps, which are calibrated using normal patch-distance statistics and selectively fused. The maximum of the fused map gives the image-level anomaly score. Across seven few-shot settings on three radio frequency (RF) datasets, SpectraMemAD improves the area under the receiver operating characteristic curve (AUROC) by 0.77–10.91 percentage points over the strongest compared baseline in each setting. The full model also exceeds the stronger individual branch by 0.20–2.88 points.

**Index Terms—** Spectrum anomaly detection, few-shot learning, training-free adaptation, feature memory, multi-scale features.

## I. Introduction

Spectrum anomaly detection identifies unauthorized transmissions, interference, equipment faults, and spectrum misuse by measuring deviation from a scene's normal state. Changes in noise floors and legitimate occupancy at a new site or frequency band can reduce the accuracy of existing thresholds or models. Initial deployment usually provides only a few normal spectrograms and no anomalies. We therefore study few-shot spectrum anomaly detection using only these normal samples as the target-scene reference.

Statistical detectors construct test statistics from energy, background distributions, or propagation properties [1]–[4], but require scene-specific threshold or background recalibration. Data-driven methods learn normal power-spectrum representations [5] and detect deviations through reconstruction [6]. Noise attention exploits the elevated noise floor after variational autoencoder (VAE) reconstruction of anomalous samples [7], while spatio-temporal models use prediction residuals [8]. These models require normal training data, and their detection scores depend on how anomalies affect reconstruction or prediction.

Recent spectrum methods use time-frequency attention and fingerprint distance [9], memory-enhanced distillation [10], graph modeling [11], and multisensor reverse distillation [12], but still require data to train or update model parameters. PatchCore [13] and SPADE [14] offer a useful alternative: they store normal visual features and detect deviations through nearest-neighbor search without parameter updates. However, their feature and scoring designs target industrial appearance anomalies rather than the global occupancy changes and localized time-frequency interference encountered in spectrograms. Combining global and local evidence also requires attention to their numerical scales. A branch with larger normal distances can dominate a direct sum even when it shows little deviation from its own normal reference.

Applying feature-memory detection to few-shot spectrum monitoring therefore requires adaptation from scarce normal references, joint representation of overall occupancy and local interference, and calibration of the two branches before fusion.

SpectraMemAD addresses these issues through dual-scale feature memories and fusion calibrated by normal patch-distance statistics. Our main contributions are as follows:

1. We propose a training-free framework for few-shot spectrum anomaly detection that shifts scene adaptation from parameter learning to memory learning. Features from a few normal spectrograms form the scene memory, allowing deployment with frozen pretrained encoders and no anomaly training samples.

2. We construct dual-scale spectrum memories using vision Transformer (ViT) and convolutional neural network (CNN) features. Independent nearest-neighbor searches in the two memories measure deviations in overall time-frequency structure and local texture, supplying complementary spatial anomaly score maps.

3. We design a spatial fusion rule that accounts for differences in the normal distance ranges of the two branches. Normal patch-distance statistics calibrate their responses, and a position-wise gate adds CNN evidence only where its distance is elevated relative to normal references and its calibrated response exceeds ViT's.

Across three radio frequency (RF) datasets, SpectraMemAD achieves the highest area under the receiver operating characteristic (ROC) curve (AUROC) among the compared methods in all seven few-shot settings and exceeds both individual branches.

## II. Proposed Method

### A. Problem Setting and Framework

Let $\mathcal{S}=\{x_i\}_{i=1}^{K}$ contain $K$ normal target-scene spectrograms. SpectraMemAD assigns each test spectrogram $x$ an anomaly score $s(x)$ without anomalous training samples or target-scene parameter updates. Following the standard threshold rule [9], [11],

$$
\hat{y}(x)=\mathbf{1}\{s(x)>\delta\},\tag{1}
$$

where $\hat y(x)=1$ denotes an anomaly and $\delta$ is the decision threshold. Spectrum anomalies can manifest as changes in overall time-frequency structure or as interference confined to localized regions. As shown in Fig. 1, frozen ViT and CNN encoders extract features from the normal references to construct separate memories. For each test feature, nearest-neighbor search finds the closest normal feature in the corresponding memory. The resulting distance maps are calibrated and fused at corresponding positions, and the maximum fused response gives the image-level score. Adaptation to a new scene uses its normal references to construct the memories and compute calibration statistics, while the encoders remain frozen.

![Overall framework](../../简化视觉架构图.png)

Fig. 1. Overall framework of SpectraMemAD.

### B. Spectrum-Aware Dual-Scale Feature Memory

Overall occupancy changes involve the arrangement of activity across time and frequency, whereas localized interference can alter texture within a small region. We use a frozen ViT $f_{\mathrm{V}}$ to extract overall structural features and a frozen CNN $f_{\mathrm{C}}$ to extract local texture features. Both retain spatial locations so that their anomaly responses can be combined in the same time-frequency region. For normal reference $x_i$,

$$
\mathbf{v}_i=f_{\mathrm{V}}(x_i),\qquad \mathbf{c}_i=f_{\mathrm{C}}(x_i),\qquad i=1,\ldots,K.
$$

Collecting $\mathbf{v}_i$ and $\mathbf{c}_i$ across normal references gives the feature matrices $\mathbf{F}_{\mathrm{V}}$ and $\mathbf{F}_{\mathrm{C}}$. The memories are

$$
\mathcal{M}_{\mathrm{V}}=\operatorname{Coreset}_{r}(\mathbf{F}_{\mathrm{V}}),
\qquad
\mathcal{M}_{\mathrm{C}}=\operatorname{Coreset}_{r}(\mathbf{F}_{\mathrm{C}}).\tag{2}
$$

Here, $\operatorname{Coreset}_r$ performs farthest-point selection: each step adds the feature with the largest distance to its nearest selected feature, until fraction $r$ is retained. This reduces memory storage and nearest-neighbor search cost by retaining a representative subset of normal features.

For test feature $\mathbf{u}$ and memory $\mathcal{M}=\{\mathbf{m}_j\}_{j=1}^{N}$, one-nearest-neighbor (1-NN) scoring uses cosine distance:

$$
d(\mathbf{u},\mathcal{M})
=\min_{j=1,\ldots,N}d_{\mathrm{cos}}(\mathbf{u},\mathbf{m}_j).\tag{3}
$$

Here, $d_{\mathrm{cos}}$ denotes cosine distance. A large minimum distance indicates a poor match to normal features. Arranging these distances at their spatial locations forms a patch-level anomaly score map for each branch. Section II-C calibrates and fuses these maps before computing the image-level score.

### C. Normal-Reference-Calibrated Fusion

Normal patch-distance statistics provide a reference for comparing the two branches. Within each branch, we match every normal patch feature to its nearest remaining normal feature, excluding its exact self-match. These distances form $\mathcal{R}_{\mathrm{V}}$ and $\mathcal{R}_{\mathrm{C}}$. Their medians set the normal baselines, and their interquartile ranges (IQRs) set the scales of variation.

The two distance maps are spatially aligned by interpolation onto a common grid $\Omega$. Let $V(x,q)$ and $C(x,q)$ denote their values at position $q\in\Omega$. Each position is calibrated using the corresponding branch's normal patch-distance statistics:

$$
z_{\mathrm{V}}(x,q)=\frac{V(x,q)-\operatorname{median}(\mathcal{R}_{\mathrm{V}})}
{\max\{\operatorname{IQR}(\mathcal{R}_{\mathrm{V}}),\epsilon\}},\tag{4a}
$$

$$
z_{\mathrm{C}}(x,q)=\frac{C(x,q)-\operatorname{median}(\mathcal{R}_{\mathrm{C}})}
{\max\{\operatorname{IQR}(\mathcal{R}_{\mathrm{C}}),\epsilon\}},\tag{4b}
$$

$$
p_{\mathrm{V}}(x,q)=\operatorname{sigmoid}\left(\frac{z_{\mathrm{V}}(x,q)}{T}\right),
$$

$$
p_{\mathrm{C}}(x,q)=\operatorname{sigmoid}\left(\frac{z_{\mathrm{C}}(x,q)}{T}\right).\tag{4c}
$$

Here, $\operatorname{IQR}(\mathcal{R})=Q_{75}(\mathcal{R})-Q_{25}(\mathcal{R})$, where $Q_{25}$ and $Q_{75}$ are the 25th and 75th percentiles of normal patch distances. The IQR measures the spread of the middle half of these distances. The constant $\epsilon=10^{-6}$ prevents division by zero. The sigmoid maps calibrated deviations to bounded scores, and temperature $T$ controls the smoothness of the mapping.

Let $\hat r_{\mathrm{C}}(x,q)\in[0,1]$ denote the normalized rank of $C(x,q)$ relative to $\mathcal{R}_{\mathrm{C}}$, using average ranks for ties. At each position, CNN evidence supplements the ViT score when its rank reaches $\tau$ and its calibrated deviation exceeds the ViT deviation. The fused map is

$$
g(x,q)=[p_{\mathrm{C}}(x,q)-p_{\mathrm{V}}(x,q)]_+
\,\mathbf{1}\{\hat r_{\mathrm{C}}(x,q)\geq\tau\},
$$

$$
A(x,q)=V(x,q)+\alpha\,\operatorname{IQR}(\mathcal{R}_{\mathrm{V}})\,g(x,q).\tag{5}
$$

Here, $g(x,q)$ is the gated local correction, $[u]_+=\max(u,0)$, and $\mathbf{1}\{\cdot\}$ is the indicator function. The indicator enforces the normal-reference rank threshold, while the positive difference determines the local correction. The IQR factor scales this correction in ViT distance units, and $\alpha$ controls its magnitude. The image-level anomaly score is the maximum of the fused map:

$$
s(x)=\max_{q\in\Omega}A(x,q).\tag{6}
$$

This maximum retains the strongest anomaly response after spatial fusion. We empirically set $\tau=0.8$, $T=2.5$, and $\alpha=2.5$ and keep them fixed across all datasets. Normal-feature memories and nearest-neighbor search follow PatchCore [13]; the fusion rule adapts gating ideas [15], [16] using robust statistics of normal patch distances [17].

## III. Experiments

### A. Experimental Setup

#### 1) Datasets and Evaluation Protocol

We evaluate on In-house RF, Public RF, and FedJam using only normal target-scene samples as references. Anomalous samples are reserved for testing. Table 1 lists the normal-reference settings and evaluation data.

Table 1. Datasets and evaluation protocols

| Dataset | Normal references | Evaluation data |
|---|---|---|
| In-house RF | One normal wideband observation (1-shot) | Five injected interference types |
| Public RF [18] | 1/2/4 normal spectrograms per frequency band | Five anomaly types at three intensity levels |
| FedJam [19] | 1/2/4 normal training spectrograms | 7,200 test spectrograms |

The normal references and test samples are disjoint. For In-house RF and Public RF, we compute metrics separately for each scene and interference setting, then report their unweighted mean. FedJam is evaluated on its complete test set.

In-house RF contains normal recordings from four campus scenes with injected burst, chirp, direct-sequence spread-spectrum (DSSS), pulse, and deceptive interference. Chirp and pulse anomalies have been evaluated in spectrum anomaly detection [8], and DSSS signals superimposed on normal transmissions are studied in ICARUS [20]. For burst, chirp, DSSS, and pulse interference, we express intensity using the jammer-to-signal ratio (JSR):

$$
\operatorname{JSR}_{\mathrm{dB}}=10\log_{10}(P_{\mathrm{J}}/P_{\mathrm{S}}),
$$

where $P_{\mathrm{J}}$ and $P_{\mathrm{S}}$ denote the interference and normal-signal powers, respectively. We empirically set the injected interference levels according to the background noise levels in the RF recordings. The In-house RF JSRs are $-10/-20/-30$ dB for burst/chirp/DSSS and $-20/-30/-40$ dB for pulse; deceptive transmit power ranges from $-30$ to $20$ dB. Public RF uses $-30/-40/-50$ dB for burst/DSSS/pulse, $-40/-50/-55$ dB for chirp, and $-10/-20/-30$ dB for deceptive signals. FedJam provides measured over-the-air jamming data rather than synthetic intensity settings.

#### 2) Baselines and Metrics

We compare eight baselines covering statistical detection, spectrum anomaly detection, and visual anomaly detection. The statistical baselines are energy detection (ED) [1] and the support-calibrated spectral ensemble (SCSE). SCSE calibrates energy, spectral entropy, spectral flatness, spectral kurtosis, and cell-averaging constant false alarm rate (CA-CFAR) statistics using normal references, then takes the largest positive standardized deviation. The spectrum baselines are IAD-PER [6], UDMA [10], GRETEL [11], and TFAM-AAE [9]; the visual baselines are PatchCore [13] and SPADE [14]. All methods use the same normal-reference and test splits.

We report AUROC, area under the precision-recall curve (AUPRC), and false-positive rate (FPR) at 95% true-positive rate (TPR), denoted by FPR@95%TPR. All metrics are expressed as percentages. Higher AUROC and AUPRC indicate better ranking performance, while lower FPR@95%TPR indicates fewer false alarms at high recall.

#### 3) Implementation

SpectraMemAD uses frozen OpenCLIP ViT-B/16-plus-240 [21] and ImageNet-pretrained ResNet18 [22] encoders. The global branch combines ViT feature layers 1–2, and the local branch uses ResNet stage 3. Each memory retains 50% of normal features through farthest-point selection and uses 1-NN search. The two patch-distance maps are calibrated and fused at corresponding spatial positions; the image-level score is their fused map's maximum. We evaluate the ViT-only and CNN-only variants to assess the contribution of combining the branches.

### B. Main Results and Analysis

#### 1) In-house RF

Table 2. Main results on In-house RF (%)

| Method | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|
| ED | 44.47 | 29.00 | 91.63 |
| SCSE | 74.89 | 56.99 | 65.63 |
| IAD-PER | 53.00 | 35.72 | 79.48 |
| UDMA | 57.95 | 38.77 | 75.76 |
| GRETEL | 63.96 | 43.80 | 68.61 |
| TFAM-AAE | 59.50 | 40.16 | 73.60 |
| PatchCore | 84.38 | 70.17 | 55.94 |
| SPADE | 71.32 | 53.16 | 66.73 |
| **SpectraMemAD** | **91.47** | **81.43** | **24.59** |

On In-house RF, SpectraMemAD achieves the highest AUROC and AUPRC and the lowest FPR@95%TPR among the evaluated methods (Table 2). Compared with PatchCore, the strongest baseline on all three metrics, it improves AUROC and AUPRC by 7.09 and 11.26 percentage points, respectively, and reduces FPR@95%TPR by 31.35 percentage points. These results show gains in both overall ranking and false-alarm control at 95% recall.

![Detection performance on In-house RF](../../analysis_outputs/20260911_inhouse_metric_triptych/inhouse_metrics_formal.png)

Fig. 2. Detection performance on In-house RF.

Table 3. Main results on Public RF (%)

| Method | 1-shot AUROC | 2-shot AUROC | 4-shot AUROC | 1-shot AUPRC | 2-shot AUPRC | 4-shot AUPRC | 1-shot FPR@95%TPR | 2-shot FPR@95%TPR | 4-shot FPR@95%TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED | 49.39 | 49.39 | 49.39 | 11.27 | 11.27 | 11.27 | 96.92 | 96.92 | 96.92 |
| SCSE | 60.05 | 61.16 | 61.34 | 9.06 | 9.23 | 9.30 | 85.05 | 82.11 | 83.58 |
| IAD-PER | 52.89 | 52.70 | 51.80 | 9.11 | 8.37 | 8.08 | 90.80 | 91.26 | 91.46 |
| UDMA | 52.15 | 52.20 | 51.91 | 9.06 | 9.03 | 9.56 | 92.52 | 92.73 | 91.99 |
| GRETEL | 57.30 | 57.89 | 57.27 | 12.78 | 13.06 | 13.72 | 88.90 | 88.46 | 87.57 |
| TFAM-AAE | 54.31 | 53.70 | 54.73 | 9.15 | 7.94 | 7.55 | 89.03 | 88.22 | 88.42 |
| PatchCore | 70.83 | 74.17 | 78.70 | 25.13 | 30.50 | 33.18 | 66.90 | 62.85 | 56.78 |
| SPADE | 66.07 | 65.78 | 67.35 | 16.72 | 17.45 | 17.96 | 77.71 | 75.72 | 77.64 |
| **SpectraMemAD** | **81.74** | **82.97** | **83.74** | **44.43** | **44.43** | **45.39** | **45.69** | **44.27** | **44.12** |

Table 4. Few-shot results on FedJam (%)

| Method | 1-shot AUROC | 2-shot AUROC | 4-shot AUROC | 1-shot AUPRC | 2-shot AUPRC | 4-shot AUPRC | 1-shot FPR@95%TPR | 2-shot FPR@95%TPR | 4-shot FPR@95%TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED | 64.93 | 64.93 | 64.93 | 84.64 | 84.64 | 84.64 | 85.72 | 85.72 | 85.72 |
| SCSE | 63.51 | 68.79 | 76.52 | 84.96 | 87.38 | 90.30 | 100.00 | 86.89 | 74.89 |
| IAD-PER | 48.77 | 42.77 | 50.51 | 73.24 | 69.87 | 74.35 | 94.22 | 96.61 | 94.50 |
| UDMA | 50.45 | 46.68 | 47.68 | 73.16 | 71.29 | 71.78 | 90.28 | 92.33 | 91.61 |
| GRETEL | 45.66 | 43.14 | 42.28 | 71.78 | 70.55 | 70.33 | 95.46 | 97.04 | 96.76 |
| TFAM-AAE | 46.43 | 43.90 | 41.83 | 72.35 | 70.78 | 69.93 | 94.67 | 94.33 | 96.00 |
| PatchCore | 81.27 | 83.71 | 83.54 | 93.45 | 94.12 | 94.13 | **74.22** | **64.06** | **67.33** |
| SPADE | 65.71 | 62.88 | 64.68 | 86.65 | 84.15 | 85.23 | 88.89 | 88.72 | 86.50 |
| **SpectraMemAD** | **83.71** | **84.48** | **84.80** | **94.62** | **94.76** | **94.75** | 78.50 | 73.22 | 68.83 |

#### 2) Public RF

On Public RF, SpectraMemAD leads on all three metrics in the 1-, 2-, and 4-shot settings (Table 3). Its AUROC exceeds PatchCore's by 10.91, 8.80, and 5.04 percentage points, respectively. Increasing the normal references from one to four raises SpectraMemAD's AUROC from 81.74% to 83.74% and lowers FPR@95%TPR from 45.69% to 44.12%. The AUROC lead is largest in the 1-shot setting and narrows as PatchCore improves with more references.

![ROC curves on Public RF](../../analysis_outputs/20260830_kshot_roc_comparison/fig_public_rf_kshot_roc.png)

Fig. 3. ROC curves of SpectraMemAD and representative baselines under different shot settings on Public RF.

#### 3) FedJam

On FedJam, SpectraMemAD achieves the highest AUROC and AUPRC in all three shot settings (Table 4). Compared with PatchCore, its AUROC gains are 2.44, 0.77, and 1.26 percentage points for 1-, 2-, and 4-shot detection, respectively. PatchCore, however, has lower FPR@95%TPR by 4.28, 9.16, and 1.50 percentage points in the same settings. Thus, SpectraMemAD improves overall ranking, while PatchCore offers better false-alarm control at 95% recall.

![ROC curves on FedJam](../../analysis_outputs/20260830_kshot_roc_comparison/fig_fedjam_kshot_roc.png)

Fig. 4. ROC curves of SpectraMemAD and representative baselines under different shot settings on FedJam.

### C. Ablation Study

Table 5. Dual-branch ablation results in AUROC (%)

| Dataset | Shot | ViT only | CNN only | SpectraMemAD |
|---|---|---:|---:|---:|
| In-house RF | 1 | 91.27 | 90.22 | **91.47** |
| Public RF | 1/2/4 | 78.86/80.97/81.62 | 75.08/75.66/75.64 | **81.74/82.97/83.74** |
| FedJam | 1/2/4 | 80.90/82.61/84.06 | 75.12/79.40/83.27 | **83.71/84.48/84.80** |

Table 5 compares the ViT-only and CNN-only variants with the full model using the same normal-reference and test splits. The ViT branch outperforms the CNN branch in all seven settings, while the full model improves AUROC over ViT by 0.20–2.88 percentage points. Thus, the local branch provides an additional gain when combined with the global branch, despite its lower standalone AUROC. These results support the joint use of global and local features for few-shot spectrum anomaly detection.

## IV. Conclusion

We have presented SpectraMemAD for few-shot spectrum anomaly detection, shifting target-scene adaptation from parameter learning to memory learning. Frozen encoders build global and local feature memories from a few normal spectrograms. Normal patch-distance statistics guide spatial fusion, and the maximum fused response determines the image-level score. Across seven few-shot settings on three datasets, SpectraMemAD improves AUROC over the strongest baseline in each setting by 0.77–10.91 percentage points. Combining the branches further improves AUROC over the stronger individual branch by 0.20–2.88 percentage points. Future work will focus on reducing false alarms at high recall, motivated by the results on FedJam.

## References

1. H. Urkowitz, "Energy Detection of Unknown Deterministic Signals," *Proceedings of the IEEE*, vol. 55, no. 4, pp. 523–531, 1967.
2. H. Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target Situations," *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-19, no. 4, pp. 608–621, 1983.
3. M. Afgani, S. Sinanovic, and H. Haas, "The Information Theoretic Approach to Signal Anomaly Detection for Cognitive Radio," *International Journal of Digital Multimedia Broadcasting*, vol. 2010, article 740594, 2010.
4. S. Liu, Y. Chen, W. Trappe, and L. J. Greenstein, "ALDO: An Anomaly Detection Framework for Dynamic Spectrum Access Networks," *IEEE INFOCOM*, pp. 675–683, 2009.
5. S. Rajendran, W. Meert, V. Lenders, and S. Pollin, "Unsupervised Wireless Spectrum Anomaly Detection With Interpretable Features," *IEEE Transactions on Cognitive Communications and Networking*, vol. 5, no. 3, pp. 637–647, 2019.
6. Y. Tian, H. Liao, J. Xu, Y. Wang, S. Yuan, and N. Liu, "Unsupervised Spectrum Anomaly Detection Method for Unauthorized Bands," *Space: Science & Technology*, vol. 2022, article 9865016, 2022.
7. J. Xu, Y. Tian, S. Yuan, and N. Liu, "Noise Attention Based Spectrum Anomaly Detection Method for Unauthorized Bands," arXiv:2104.08517, 2021.
8. C. Peng, W. Hu, and L. Wang, "Spectrum Anomaly Detection Based on Spatio-Temporal Network Prediction," *Electronics*, vol. 11, no. 11, article 1770, 2022.
9. H. Ji, T. Zhang, X. Qiao, H. Wu, and G. Gui, "TFAM-AAE-U$_k$: A Dual-Metric Spectrum Anomaly Detection Algorithm," *IEEE Communications Letters*, vol. 28, no. 11, pp. 2638–2642, 2024.
10. P. Qi, T. Jiang, J. Xu, J. He, S. Zheng, and Z. Li, "Unsupervised Spectrum Anomaly Detection With Distillation and Memory Enhanced Autoencoders," *IEEE Internet of Things Journal*, vol. 11, no. 24, pp. 39361–39374, 2024.
11. A. Hussain, T. Hussain, K. I. Rashid, et al., "GRETEL: A Graph Attention Network for Low-SNR Spectrum Anomaly Detection in IoT Communications," *IEEE Internet of Things Journal*, vol. 13, no. 17, pp. 38628–38641, 2026.
12. R. Zhang, T. Zhang, H. Wu, G. Ding, H. Ji, and J. Zhang, "Detection and Localization of Spectrum Anomaly with Multi-Sensors: A Memory-Augmented Reverse Distillation Method," *IEEE Internet of Things Journal*, 2026, doi: 10.1109/JIOT.2026.3717688.
13. K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, "Towards Total Recall in Industrial Anomaly Detection," *CVPR*, pp. 14318–14328, 2022.
14. N. Cohen and Y. Hoshen, "Sub-Image Anomaly Detection with Deep Pyramid Correspondences," arXiv:2005.02357, 2020.
15. R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive Mixtures of Local Experts," *Neural Computation*, vol. 3, no. 1, pp. 79–87, 1991.
16. J. Arevalo, T. Solorio, M. Montes-y-Gómez, and F. A. González, "Gated Multimodal Units for Information Fusion," *ICLR Workshop*, 2017.
17. P. J. Rousseeuw and M. Hubert, "Anomaly Detection by Robust Statistics," *WIREs Data Mining and Knowledge Discovery*, vol. 8, no. 2, e1236, 2018.
18. M. Wellens and P. Mähönen, "Lessons Learned from an Extensive Spectrum Occupancy Measurement Campaign and a Stochastic Duty Cycle Model," *Mobile Networks and Applications*, vol. 15, no. 3, pp. 461–474, 2010.
19. I. Panitsas, I. Ofeidis, and L. Tassiulas, "FedJam: Multimodal Federated Learning Framework for Jamming Detection," *IEEE INFOCOM*, pp. 1–10, 2026.
20. D. Roy, V. Chaudhury, C. Tassie, C. Spooner, and K. R. Chowdhury, "ICARUS: Learning on IQ and Cycle Frequencies for Detecting Anomalous RF Underlay Signals," *IEEE INFOCOM*, pp. 1–10, 2023, doi: 10.1109/INFOCOM53939.2023.10228929.
21. M. Cherti, R. Beaumont, R. Wightman, et al., "Reproducible Scaling Laws for Contrastive Language-Image Learning," *CVPR*, pp. 2818–2829, 2023.
22. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, pp. 770–778, 2016.
