# `PromptAD/model.py` 通俗导读

> 目标读者：第一次进入这份代码、对 CLIP / 异常检测细节不熟悉的人。
>
> 文件 3131 行，看起来很吓人。但其实只有 **4 大块**，剩下都是这 4 块的变体。
> 看完这份文档，你应该能直接按章节回到 `model.py` 对应位置读代码。

---

## 0. 这份代码到底在干嘛？

一句话：**用一个 CLIP 模型（图像-文本对齐的预训练模型），从一堆"正常的频谱图"里学会判断哪张是"异常"。**

它判断异常用了 **两条腿**：

1. **文本腿**：把图像送进 CLIP 图像编码器拿到一个向量，再把"正常 / 异常"两条文字 prompt 也送进 CLIP 文本编码器拿到两个向量。
   图像向量跟谁更像，就更可能是哪一类。
2. **视觉腿**：把训练用的正常图像的每个小块（patch）特征都存下来，做一个"正常 patch 字典"。
   测试时新图像的每个 patch 去字典里找最近的，找不到像的就是"异常 patch"。

文本腿用来出**图像级**异常分数（这张图是不是异常）；视觉腿主要出**像素级**异常图（图像中哪一块异常）。

`model.py` 这一个文件做的事就是：
- 把图像变成 CLIP 能吃的张量（**预处理 transform**）；
- 调用 CLIP 算特征（**encode_image / encode_text**）；
- 准备 prompt（**PromptLearner**）；
- 用上面两条腿算分数（**calculate_textual_anomaly_score / calculate_visual_anomaly_score**）；
- 各种"想让结果更好"的实验性扩展（adapter / VCPA / RN50 / CNN-Mamba / dense head…）。

---

## 1. 文件的 4 大块

打开 `model.py`，从上往下大致是这样：

| 行号区间（大致） | 章节 | 干啥的 |
|---|---|---|
| `1 – 36` | **顶部导入和常量** | 导入 PyTorch、CLIP 封装、定义 CLIP 的 mean/std |
| `38 – 1304` | **一大堆 channel transform 类** | 把灰度频谱图变成 3 通道彩色图的各种"花式滤镜"。每个类都长得很像，只是滤镜不同 |
| `1306 – 1538` | **可学习的小模块（adapter / fusion 头 / 局部分支）** | 5 个轻量模块，CLIP 主干冻结时只训练它们 |
| `1541 – 3131` | **核心：`PromptLearner` + `PromptAD`** | 真正的模型类，负责出异常分数 |

下面挨个讲。

---

## 2. 第一块：图像预处理 transform

### 它为什么存在？

CLIP 是在自然图片（猫、汽车、风景）上预训练的，它期望的输入是 **3 通道 RGB**。
但我们手里的是**频谱图**——本质上是个灰度二维数组，单通道。
直接把灰度图复制 3 份当 RGB 输入也行，但会损失很多信息。

所以这部分代码本质上回答一个问题：
> "灰度频谱图怎么变成 3 通道，才能既骗过 CLIP，又凸显信号异常？"

每个 `*Channels` 类都是一种**灰度 → RGB**的变换方案。结构都长一样：

```python
class XxxChannels:
    def __init__(self, ...):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))   # 拿到灰度
        # ... 算几个"派生通道"（梯度图、残差图、CLAHE 增强图……）
        return torch.cat([通道1, 通道2, 通道3], dim=0)  # 拼成 3 通道
```

只要你看懂这个套路，剩下 30 多个类都是**换不同的派生通道**而已：

| 类名（举几个例子） | 三个通道分别是啥 | 直观意义 |
|---|---|---|
| `SpectrogramGradientChannels` | 灰度 / 时间方向梯度 / 频率方向梯度 | 强调变化剧烈的位置 |
| `MorphFusionChannels` | 灰度 / 梯度幅值 / 弱残差 | 边缘 + 偏离背景 |
| `MorphFusionGrayResidualChannels` | CLAHE 对比度增强灰度 / 弱残差 / 原始灰度 | **当前主方案**：低对比信号更显眼 |
| `ChirpDirectionalChannels` | 灰度 / 梯度 / 对角线响应 | 给 chirp（线性调频）信号用，强调斜线 |
| `DSSSStatisticalChannels` | 灰度 / 残差 / 局部方差 | 给 DSSS 信号用，强调局部能量 |
| `NormalBgDeviationChannels` | 像素相对训练正常背景的偏离倍数（重复 3 份） | 用统计的眼光看"这一点不正常" |

> ⚠️ 这里出现的术语简单解释：
> - **CLAHE**：一种局部对比度增强算法（OpenCV 直接提供），把暗的地方拉亮、亮的地方压暗。
> - **残差**：原图 - 背景估计，凸显"偏离背景的部分"。这里背景常用"每一行的中位数"近似。
> - **Gabor 核**：一种带方向、带频率的滤波器，能检测特定方向的纹理。
> - **Sobel**：经典的边缘检测算子。

**实践上你不需要每个类都看懂**。在 `PromptAD.__init__` 里有个长 `if/elif` 链（约 1905 – 2089 行），根据 `--input_mode` 选择用哪个变换。
当前主方案是 `morph_fusion_gray_residual_a01`（对应 `MorphFusionGrayResidualChannels(alpha=0.1)`）。

---

## 3. 第二块：可学习的小模块

CLIP 主干很大（千万级参数），重新训练成本高、也容易过拟合到几张正常样本。
所以这份代码采取的策略是：**冻结 CLIP，只在外面加几个轻量小模块**，用很少的参数学到任务相关的修正。

这 5 个模块都是可选的（命令行加 `--xxx` 才会启用）。它们有一个共同的小技巧：
**最后一层权重初始化为 0**，这样训练刚开始时小模块输出全是零，相当于"什么都没做"，不会破坏 CLIP 原本的能力；
然后训练过程中梯度才慢慢让它学到一些有用的修正。

| 模块 | 行号 | 输入 → 输出 | 直观作用 |
|---|---|---|---|
| `ResidualVisualAdapter` | 1306 | 视觉特征 → 视觉特征 + 小修正 | 让 CLIP 视觉特征更适配频谱任务 |
| `VisualClassPromptAdapter`（VCPA） | 1338 | 正常图像原型 → 一组"软 class token" | 让 prompt 中的"类别词"也能受图像驱动 |
| `LearnableScoreFusionHead` | 1378 | 4 个分数标量 → 一个修正值 | 学一个"是否相信文本分数"的小裁判 |
| `DenseMaskHead` | 1404 | 两路 patch 特征 → 像素级 logits | 当有真值 mask 时做像素监督 |
| `TinyMambaFusionBlock` + `TinyCNNPatchEncoder` + `CNNMambaLocalBranch` | 1438 / 1479 / 1512 | 图像 → patch token 序列 | 一个完全独立于 CLIP 的局部异常分支 |

> 💡 **新手只需要记住**：这部分都是"可选的额外模块"，没启用的时候它们完全不参与计算。
> 第一次读代码可以先跳过，等读懂主流程再回来看。

---

## 4. 第三块：`PromptLearner`（约 1541 行起）

### 这一段在解决什么问题？

我们要让 CLIP 文本编码器输出"正常频谱"和"异常频谱"两个文本向量。
最朴素的做法是直接写一句英文：`"a normal radio frequency spectrum."`，但效果有限。

**PromptAD 论文的思路是**：让一部分单词变成**可训练的向量**。具体长这样：

```
正常 prompt：   [SOS]  N  N  N  N  radio_frequency_spectrum  .  [EOS]
                       ↑ ↑ ↑ ↑
                       这 4 个位置是可学习向量 normal_ctx

可学习异常 prompt： [SOS]  N N N N  A A A A  radio_frequency_spectrum  .  [EOS]
                          ↑─正常上下文   ↑─异常上下文
                          normal_ctx   abnormal_ctx
```

`N` 和 `A` 只是占位符——CLIP 会先把整句话 tokenize 成 token id，再 embedding 成向量；
然后我们**手动把 N/A 位置对应的 embedding 替换成可训练的 `normal_ctx` / `abnormal_ctx`**。

训练的时候，梯度通过 CLIP 文本编码器回传，更新这些向量，让它们逐渐学到"什么样的方向代表正常 / 异常"。

### 还有第二种异常 prompt

代码里同时维护了**两种**异常 prompt：

1. **手工 prompt**（`abnormal_prompts_handle`）：用 `ad_prompts.py` 里写好的英文模板，例如：
   - `"a damaged spectrum"`
   - `"a flawed spectrum"`
   - `"a spectrum with defect"`
   它们的可训练部分还是 `normal_ctx`，异常语义来自手写的英文词。
2. **可学习 prompt**（`abnormal_prompts_learned`）：用上面提到的 `abnormal_ctx` 真正学异常方向。

两者各有 prompt 数量（`n_pro` 决定每条模板复制几份，`n_pro_ab` 决定可学习异常 prompt 有几条）。
最后做异常文本特征时，把它们的特征**全部平均**成一个"异常原型"向量（除非用了 grouped 模式）。

### `forward()` 输出什么？

`PromptLearner.forward()` 不直接出文本特征，而是出 **prompt 的 embedding 序列**：

```
normal_prompts          : (n_pro, seq_len, dim)        正常 prompt 的 embedding
abnormal_prompts_handle : (n_pro * n_ab_handle, …)     手工异常 prompt
abnormal_prompts_learned: (n_pro * n_pro_ab,    …)     可学习异常 prompt
```

这些 embedding 之后会被送进 CLIP 文本 encoder（在 `PromptAD.build_text_feature_gallery` 里），才变成最终的 1 维向量。

---

## 5. 第四块：`PromptAD`（约 1804 行起，文件的主类）

这是真正"对外"的模型类。它把 CLIP、PromptLearner、各种缓存、各种 adapter 装在一起。

### 5.1 `__init__` 做了 4 件事

1. **解析超参**：从 `kwargs` 里读一大堆开关（约 1838–1885 行），决定要不要启用 adapter、VCPA、RN50、CNN-Mamba 等可选模块；
2. **构建模型**：调用 `self.get_model()` 创建 CLIP 主干、PromptLearner、可选模块、各种 buffer（gallery / 文本特征）；
3. **选预处理**：根据 `input_mode` 选一个 channel transform（就是第二块讲的"灰度 → RGB"），拼成 `self.transform`；
4. **mask 预处理**：另起一条 `self.gt_transform` 给标签 mask 用（用 NEAREST 插值，避免插出非 0/1 值）。

### 5.2 内部维护的缓存（"gallery"）

这是 PromptAD 的精髓之一——它把训练时的"标准答案"存成一些 buffer：

| 名字 | 形状 | 内容 |
|---|---|---|
| `text_features` | (2, dim) 或 (1+组数, dim) | 正常 / 异常文本原型（PromptLearner 训练后 build） |
| `feature_gallery1`, `feature_gallery2` | (shot × patch数, dim) | 正常训练图像两路 patch 特征字典 |
| `cnn_mamba_gallery` / `cnn_mamba_global_gallery` | 同上 / (shot, dim) | CNN-Mamba 局部分支用的字典 |
| `rn50_gallery` / `rn50_local_gallery` | (shot, dim) / 类似 | RN50 副分支用的字典 |
| `normal_global_mean / center / var` | (dim,) | 正常全局特征的统计量，给"高斯/马氏距离"分数用 |

它们都用 `register_buffer` 注册——意思是"算模型状态、跟着 .to(device) 走，但不参与梯度更新"。

### 5.3 训练-测试核心流程

```
                    train_cls.py（外面的训练脚本）
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
[1] 跑一遍 train_loader   [2] 训练 epoch       [3] 测试 epoch
    encode_image
    build_image_feature_gallery   ─循环─►   forward / score_cached
    （把正常 patch 存进 gallery）            ◄──────────────────
                                build_text_feature_gallery
                                （从 PromptLearner 重算文本原型）
```

具体每个函数的角色：

| 函数 | 职责 |
|---|---|
| `encode_image(images)` | CLIP 图像编码 → 4 路视觉特征，全部 L2 归一化 |
| `encode_text_embedding(emb, tokens)` | 把 PromptLearner 的 embedding 序列送进 CLIP 文本 encoder |
| `build_image_feature_gallery(...)` | 训练开始时把正常样本的 patch 特征塞进 `feature_gallery1/2` |
| `build_text_feature_gallery()` | 每个 epoch 结束后，根据当前 `normal_ctx/abnormal_ctx`，重新生成 `text_features` |
| `calculate_textual_anomaly_score(...)` | **文本腿**：图像特征 × 文本原型 → softmax → 异常概率 |
| `calculate_visual_anomaly_score(...)` | **视觉腿**：每个 patch 找最近的正常 patch，距离 = 异常分数 |
| `forward(images, task)` | 完整推理入口：cls 出图像分数 + patch 图；seg 出像素图 |
| `score_cached(visual_features, ...)` | 复用已缓存的视觉特征，只重新跑文本部分（评估阶段省时间） |

### 5.4 训练的 loss 在哪里？

`model.py` 本身**不写 loss**，它只暴露"算分数"的接口。
真正的 loss（vision-to-text 交叉熵、triplet loss、prompt consistency loss）写在 `train_cls.py` / `train_seg.py` 里，调用模型的中间产物来组合。
这是一种"模型只管特征、训练脚本管 loss"的解耦写法。

### 5.5 `cls_score_mode`：图像级分数怎么得到？

`fuse_cls_scores()` 是图像级分数的总入口（约 2948 行），由 `--cls-score-mode` 控制：

- `text_only`：直接用文本异常分数（**主方案**，最简单）；
- `visual_topk`、`visual_topk_max`：文本分 + 视觉 patch 图 top-k 均值；
- `normal_center` / `normal_mahalanobis`：完全不用文本，只看离正常中心多远；
- `text_normal_*`：上面两者的加权和；
- `dense_only` / `dense_fusion`：用 DenseMaskHead 替换或叠加视觉腿。

### 5.6 三个"附加"打分通路

它们不走 `fuse_cls_scores`，而是在 `score_cached` 里**直接加到 image_scores 上**：

1. **RN50 副分支**（`--rn50-visual-fusion`）：另跑一个 CLIP-RN50 当独立"参考意见"；
2. **CNN-Mamba 局部分支**（`--cnn-vit-mamba-fusion`）：可训练的 CNN+Mamba 序列；
3. **可学习融合头**（`--learnable-score-fusion`）：拿 (text 分数, visual top-k 均值, max, gap) 4 个标量喂小 MLP，输出一个修正值。

新手看代码可以先**跳过这 3 个**，主流程跑通后再回来。

---

## 6. 一张图把它们串起来

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 数据流入                                                      │
│   图像（频谱图） ── self.transform（第 2 块）─► 3 通道张量        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CLIP 视觉编码（冻结，可选 adapter / LoRA）                    │
│   encode_image → [cls, tokens, patch_map1, patch_map2]           │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┴────────────────────┐
            ▼                                        ▼
   ┌───────────────────┐                  ┌─────────────────────┐
   │ 3a 文本腿         │                  │ 3b 视觉腿            │
   │                   │                  │                     │
   │ PromptLearner ──► │                  │ feature_gallery1/2  │
   │ build_text_feat   │                  │ ◄──── 训练时填入    │
   │ → text_features   │                  │                     │
   │                   │                  │ 每个 patch 找最近邻 │
   │ image · text^T    │                  │ 1 - cosine = 距离   │
   │ softmax → 异常率  │                  │ → patch 异常图      │
   └─────────┬─────────┘                  └──────────┬──────────┘
             │                                       │
             ▼                                       ▼
       图像级分数                               像素级分数
       （cls，Image-AUROC）                    （seg，Pixel-AUROC）
                                                     │
                                  fuse_cls_scores ◄──┘ 文本+视觉融合
```

---

## 7. 读代码时的推荐顺序

第一次读这份文件，建议按以下顺序：

1. **看 `__init__` 的开头**（`PromptAD.__init__`，约 1804–1900 行）：先了解模型构造参数；
2. **看 `forward`**（约 3052 行）：理解一次推理的总流程；
3. **看 `calculate_textual_anomaly_score`**（约 2752 行）：理解文本腿；
4. **看 `calculate_visual_anomaly_score`**（约 2853 行）：理解视觉腿；
5. **回头看 `PromptLearner`**（1541 行）：理解 prompt 是怎么学的；
6. **看 `build_text_feature_gallery`**（约 2655 行）：把 1–5 串起来；
7. **看 channel transform**（第 2 块）：挑当前 input_mode 对应的那一个看就行；
8. **可选模块**：等主流程理解了再回来看。

---

## 8. 一份术语速查

| 术语 | 通俗解释 |
|---|---|
| **CLIP** | OpenAI 提出的图像-文本对齐模型。一个图像编码器 + 一个文本编码器，能把图像和文字映射到同一个向量空间 |
| **ViT** | Vision Transformer，CLIP 的图像编码器之一。把图像切成 14×14 或 16×16 个 patch，再过 transformer |
| **patch** | ViT 把图像切成的小方块，每个 patch 对应一个特征向量 |
| **token** | 文本/图像被切分后的最小单元。文本 token 是 BPE 子词；图像 token 就是 patch |
| **embedding** | 把离散 id 变成连续向量。"embedding 序列"就是"一串向量" |
| **prompt** | 给 CLIP 的提示文本，例如 `"a photo of a cat"`。在 PromptAD 里，prompt 中一部分单词被换成可训练向量 |
| **gallery / memory bank** | 一个"参考样本字典"，测试时新样本去字典里找最近的 |
| **L2 归一化** | 把向量除以自己的长度，让长度变成 1。归一化之后，两个向量的内积就等于余弦相似度 |
| **logit_scale** | CLIP 的温度参数，乘在相似度上让 softmax 更尖锐 |
| **AUROC** | ROC 曲线下面积，一个常用的二分类评估指标（0.5=瞎猜，1.0=完美） |
| **patch 异常图** | 一张 H×W 的图，每个位置一个分数，越高越像异常 |
| **few-shot** | 每类只用很少几个样本训练（这里 k-shot=1 表示每类只用 1 张正常图） |
