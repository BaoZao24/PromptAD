# AdaptCLIP 中的 Adapter 设计：一份可读源码级教程

本文档系统地讲解 **AdaptCLIP** 项目里 adapter 的整体思路、三种 adapter 的具体结构、以及它们如何配合冻结的 CLIP 主干完成「零样本 / 小样本」工业异常检测。

阅读完之后，你应该能回答这些问题：

1. 为什么要在 CLIP 之外再加 adapter？
2. 项目里有几个 adapter，各自负责什么？
3. 这些 adapter 是怎么连进 CLIP forward 流程的？
4. 训练时哪些参数被冻结、哪些参数在更新？
5. 推理时多个 adapter 的输出是怎么融合成一张异常图和一个异常分数的？

所有源码引用都附带 `file:line` 形式的位置，方便对照阅读。

---

## 1. 设计哲学：冻结 CLIP，只训练「轻量适配器」

CLIP 在大规模图文对上预训练，已经有很强的「正常 / 异常」语义判别能力，但它：

- 没见过工业缺陷的细粒度分布；
- 视觉编码主要服务于 *图级* 对齐，对 *像素级* 异常定位不友好；
- 文本端只有手写模板，没法针对异常检测任务去 fine-tune prompt。

AdaptCLIP 的策略是：

> **CLIP 主干完全冻结**（包括 visual encoder 和 text encoder），只训练三个轻量的 adapter，把 CLIP 的特征"扳"到异常检测任务上。

可以从 `train.py:73-76` 看到这点：

```python
model.eval()             # CLIP 主干 eval，且 forward 时全程包在 torch.no_grad() 里
textual_learner.train()
visual_learner.train()
pq_learner.train()
```

而 `train.py:131-133` 中的 `model.encode_image(...)` 就被 `torch.no_grad()` 包着，主干根本不参与反向传播。优化器（`train.py:95-99`）也只把三个 adapter 的参数喂给 Adam。

---

## 2. 在动 adapter 之前：先看 CLIP 主干被改了什么

虽然主干"冻结"，但作者还是对 ViT 的最后几层做了一个**结构改造**——这是后续所有 adapter 能 work 的前提。

### 2.1 V-V Attention（DPAM）

CLIP ViT 默认是 Q·K·V 自注意力，输出的 patch token 会受到 [CLS] 的牵引，**不利于像素级定位**。AdaptCLIP 沿用了 SAA、AnomalyCLIP 一脉的做法：把最后若干层（默认 20 层中的最后 20 层都换、即 `DPAM_layer=20`）的注意力替换成 **V-V self-attention**——`q = k = v`，让每个 patch 只跟在视觉上最相似的 patch 交互。

代码在 `adaptclip.py:105-141` 的 `Attention` 类，注意它会同时输出：

- `x_ori`：走原始 Q·K·V 路径的结果（保留全局语义，给 [CLS] 用）
- `x`：走 V-V 路径的结果（给 patch 定位用）

替换是在 `VisionTransformer.DAPM_replace` 里完成的（`adaptclip.py:396-405`）：把最后 `DPAM_layer` 层 `nn.MultiheadAttention` 的权重 clone 进新的 `Attention` 模块。所以推理时 `model.encode_image` 返回的是 *改造后的特征*，但权重数值跟原始 CLIP 完全一致。

### 2.2 多层级 patch token 输出

`Transformer.forward`（`adaptclip.py:347-359`）会按 `out_layers=[6, 12, 18, 24]` 收集每一层的 patch token 输出。这意味着 adapter 拿到的是 **4 个尺度** 的 patch 特征，而不是只有最后一层——浅层利于细粒度定位，深层利于语义判别。

---

## 3. 三个 Adapter 概览

| Adapter | 对应 `train.py` 开关 | 输入 | 输出 | 训练参数量级 |
|---|---|---|---|---|
| `TextualAdapter` | `--textual_learner` | learnable text prompt | (image-level logit, pixel map) | ~10K（CoOp 风格） |
| `VisualAdapter` | `--visual_learner` | query 的 image / patch 特征 | (image-level logit, pixel map) | ~0.1M |
| `PQAdapter` | `--pq_learner` | query + few-shot prompt 特征 | (image logits, pixel maps, align maps) | ~1M |

**关键观察**：三个 adapter 是**独立的、可单独开关的**「分支」，最终输出在推理阶段被融合（harmonic / arithmetic mean）。它们三者各自从不同角度刻画"正常 vs 异常"。

下面逐个讲。

---

## 4. TextualAdapter：把 text prompt 变成可学习的

源码：`adaptcliplib/adaptclip.py:846-1042`。

### 4.1 它在做什么

CLIP 做异常检测的"零样本基线"长这样：

```
text_normal   = "a photo of a flawless object."
text_anomaly  = "a photo of a damaged object."
text_features = encode_text([text_normal, text_anomaly])
score = softmax(image_feat @ text_features.T)[..., 1]
```

但 "flawless / damaged" 这种词到底是不是工业领域最好的提示？不一定。`TextualAdapter` 把 prompt 拆成两部分：

1. **静态模板** (`static_normal_list` + `template_list`，`adaptclip.py:866-905`)
   一堆人工写的句子（"a cropped photo of the {}."、"flawless {}" 等等），共 `7 × 21 = 147` 条 normal、`4 × 21 = 84` 条 anomaly，编码后取均值得到 `static_text_features`（`prepare_static_text_feature`，`adaptclip.py:1012-1023`）。这部分**不参与训练**，但被 `VisualAdapter` 当作"语义锚点"用。

2. **可学习上下文向量** (`ctx_pos`, `ctx_neg`，`adaptclip.py:913-921`)
   这是 **CoOp** 风格的做法：把 `n_ctx=12` 个 token embedding 设为 `nn.Parameter`，训练时跟随梯度更新。每次 forward (`adaptclip.py:962-1002`) 把它和固定的 `<sot>` / `<eot>` 拼成一条完整的 prompt 序列，再走一次 text encoder（`model.encode_text_learn`）得到 `learned_text_features`。

### 4.2 用学到的 prompt 算分数

`compute_global_local_score`（`adaptclip.py:1025-1042`）做两件事：

```python
# 全局：CLS token 与 [normal_text, anomaly_text] 余弦相似度 -> softmax
text_probs = (query_feats / |·|) @ text_features.T / 0.07

# 局部：最后一层 patch token 与同样两个文本特征做相似度，得到一张 H×W×2 的图
similarity, _ = compute_similarity(patch_feature, text_features[0])
similarity_map = get_similarity_map(similarity[:, 1:, :], img_size)
```

注意 `compute_similarity`（`adaptclip.py:633-638`）的相似度算的是 `(patch_feat · text_feat)`，再走 softmax 得到一个 patch 级的二分类概率分布（"是异常" / "是正常"）。

### 4.3 训练目标

在 `train.py:152-162`：

```python
global_loss += F.cross_entropy(global_logit, label)
local_loss  += loss_focal(local_score, gt)
local_loss  += loss_dice(local_score[:, 1, ...], gt)      # 异常通道 vs gt
local_loss  += loss_dice(local_score[:, 0, ...], 1-gt)    # 正常通道 vs 1-gt
```

> **要点**：可学习的只有 `ctx_pos` 和 `ctx_neg`（每个形如 `(1,1,12,768)`），加起来不到一万个参数。但因为每次都要重新过 text encoder，实际计算量并不算小。

---

## 5. VisualAdapter：给 image / patch 特征做"轻量整形"

源码：`adaptcliplib/adaptclip.py:659-712`。

### 5.1 ResMLP 基本块

```python
class ResMLP(nn.Module):
    def __init__(self, c_in, reduction=4):
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.BatchNorm1d(c_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )
    def forward(self, x):
        ...
        return x + self.fc(x)   # 残差连接
```

经典的 bottleneck MLP + residual：先把 768 维降到 192 维（`reduction=4`），再升回去。**残差使它默认是恒等映射**——一开始几乎不破坏 CLIP 原有特征，训练时再慢慢"微调"。

### 5.2 用法：分别处理 image / patch

```python
class VisualAdapter(nn.Module):
    def __init__(self, ...):
        self.local_adater  = ResMLP(input_dim, reduction)   # 处理 patch
        self.global_adapter = ResMLP(input_dim, reduction)  # 处理 [CLS]

    def forward(self, image_features, patch_features, static_text_features):
        ...
        image_feature = self.global_adapter(image_features)
        patch_feature = self.local_adater(patch_features[-1])  # 只用最后一层 patch
```

然后跟 `static_text_features`（注意：是 TextualAdapter 里那批**静态模板**生成的，**不可学习**）做一次余弦相似度，输出形状跟 TextualAdapter 完全对齐：

- `static_text_probs`：图级 logits，形状 `(B, 2)`
- `similarity_map`：像素级 anomaly 概率图，形状 `(B, 2, H, W)`

### 5.3 为什么要存在

直观理解：**TextualAdapter 在调整 *文本端*，VisualAdapter 在调整 *视觉端***。两者用的"对面"特征都来自冻结主干（一个是 static text，一个是 image / patch），只有自己这一侧在更新。把它们结合相当于"双向适配"。

参数量约 `2 × (768 × 192 + 192 × 768) ≈ 0.6M`，相比 CLIP 的 0.4B 只是九牛一毛。

---

## 6. PQAdapter：小样本时的"Prompt vs Query"对比器

源码：`adaptcliplib/adaptclip.py:715-800`。这是项目里最重要、也最有特色的 adapter。

### 6.1 动机

前两个 adapter 都是 *零样本* 风格：只看 query 自己 + 文本。但工业场景里我们常常能拿到几张 **正常样本**（few-shot），它们提供的"正常分布"其实极有信息量。`PQAdapter` 要做的就是：

> 把 query 的每个 patch 跟 prompt（=正常样本）库里最相似的 patch 比一比，差得越多越像异常。

### 6.2 输入

```python
def forward(self, query_feats,        # (B, D)        query 的 [CLS]
                  query_patch_feats,  # list[(B, L, D)]    4 层 patch
                  prompt_feats,       # (B, S, D)     S 张 few-shot 正常图的 [CLS]
                  prompt_patch_feats):# list[(B, S, L, D)]
```

`prompt_feats` 怎么来？看 `test.py:37-91` 的 `build_prompt_memory`：在测试开始前，遍历一个 `PromptDataset`（每个类别 `k_shots` 张正常图），用冻结的 `model.encode_image` 把它们的特征存进 dict，按类别索引；推理时再通过 `prompt_association`（`test.py:21-34`）把 query 对应类别的 prompt 特征取出来。训练时则在数据集 `Dataset` 里直接配对（每个 batch item 同时返回 `img` 和 `prompt_img`，见 `train.py:120-122`）。

### 6.3 内部结构（按 patch 层级展开）

`PQAdapter` 内部对 `features_list` 中每一层都各有一个 `local_adapter`（小卷积 U-Net）和 `global_adapter`（小 MLP），存在 `nn.ModuleList`（`adaptclip.py:725-751`）。**每层独立学一个分类头**，最终把多层结果平均。

每一层的 forward 流程（`adaptclip.py:756-798`）可拆成 5 步：

#### Step 1: nearest-neighbor align（找最近邻）

```python
query_patch_feat  = F.normalize(...)[:, 1:, :]        # (B, L, D)，去掉 CLS
prompt_patch_feat = F.normalize(...)[:, :, 1:, :]     # (B, S, L, D)
prompt_patch_feat = prompt_patch_feat.reshape(B, S*L, D)

align_score, min_idx = torch.min(
    1.0 - bmm(query_patch_feat, prompt_patch_feat.permute(0,2,1)),  # 余弦距离
    dim=-1
)                                                     # 对每个 query patch 找最近的 prompt patch
```

`align_score` 直接就是一个**「无参数」的异常图**——每个 query patch 和最相似的正常 patch 的距离。它会被 `F.interpolate` 上采样到原图分辨率，作为 `align_scores` 的一个元素，最终也参与融合。

#### Step 2: 取出对齐到的 prompt patch

```python
align_prompt_feat = prompt_patch_feat[arange(B).unsqueeze(1), min_idx]  # (B, L, D)
```

这是经典的 "gather by index"：query 的第 `i` 个 patch 拿到的是它最像的那个 prompt patch 的特征。

#### Step 3: 构造 fusion 特征（残差 / 绝对差）

```python
if self.context:
    fusion_patch_feat = query_patch_feat + |query_patch_feat - align_prompt_feat|
else:
    fusion_patch_feat = |query_patch_feat - align_prompt_feat|
```

`context=True` 时多保留一份 query 自己的语义；`context=False` 则纯粹看 "差异"。这部分由 `--pq_context` 控制。然后过一个 BatchNorm（`sharebn[lay_idx]`，`adaptclip.py:725, 785`）。

> 这一步是 PQAdapter 的核心 idea：把 **「跟最近邻正常样本的差」** 当作分类器的输入，让 conv head 去学"哪些差异是真的异常"。

#### Step 4: local adapter 输出像素级异常图

```python
local_score = local_adapter[lay_idx](fusion_patch_feat)   # (B, 2, h', w')
local_score = local_score.softmax(dim=1)
local_score = F.interpolate(local_score, (img_size, img_size), 'bilinear')
```

`local_adapter` 是一段 conv + 两次 ConvTranspose2d 的小 U-Net（`adaptclip.py:727-737`），把 `H/14 × W/14` 的特征图渐进上采到 `H/4 × W/4` 再上插值到全图。

#### Step 5: global adapter 输出图级 logit

```python
fusion_img_feat = fusion_patch_feat.view(B, D, -1)
fusion_img_feat = (fusion_img_feat.mean(-1) + fusion_img_feat.topk(10,-1)[0].mean(-1)) / 2.0
global_logit = global_adapter[lay_idx](fusion_img_feat)
```

特别注意**池化方式**：不是单纯 mean，而是 `mean + top-10-mean`。这样既保留全局信息，又强调最异常的 10 个 patch——对"小区域缺陷会被平均掉"的问题非常关键。

### 6.4 训练目标

`train.py:164-174` 对每一层都算一遍 CE / focal / dice：

```python
for i in range(len(global_logit)):
    global_loss += F.cross_entropy(global_logit[i], label)
for i in range(len(local_score_list)):
    local_loss  += loss_focal(local_score_list[i], gt)
    local_loss  += loss_dice(local_score_list[i][:, 1], gt)
    local_loss  += loss_dice(local_score_list[i][:, 0], 1 - gt)
```

注意**不会**对无参数的 `align_scores` 算 loss——它只在推理阶段被融合用。

---

## 7. 推理时的融合：把三路输出"拧"成一张异常图

`test.py:280-290` 是推理融合的全部逻辑：

```python
if k_shots > 0:
    # pixel
    pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map, local_pq_map],
                                    fusion_type=args.fusion_type)            # 默认 average_mean
    pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score],
                                    fusion_type='harmonic_mean')             # 再跟无参数的 align 图融合

    pixel_anomaly_map = gaussian_filter(..., sigma=args.sigma)               # 后处理平滑

    # image
    anomaly_map_max  = pixel_anomaly_map.view(B, -1).max(dim=1)              # max pooling
    image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, global_pq_score],
                                     fusion_type=args.fusion_type)
    image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max],
                                     fusion_type='harmonic_mean')
```

关键设计：

1. **同质量分支用算术平均**：三个 adapter 的输出直接算术平均（`average_mean`）。
2. **跨质量分支用调和平均**：算术平均后的 score 再跟 `align_score`（无参数 NN 距离）以及 `anomaly_map_max`（pixel 推上来的 image score）做 **harmonic_mean**——只要其中一项很低就把整体压低，鼓励"两边都觉得异常"才报警。
3. **零样本分支退化**：`k_shots=0` 时没有 `pq_learner` 也没有 `align_score`，融合退化为只有 vl + tl 两项。

`harmonic_mean` 和 `average_mean` 的实现在 `adaptclip.py:589-622`，没什么花里胡哨的，就是字面意思。

---

## 8. 数据流总览（一张图代替一千字）

```
                ┌─── encode_image (frozen, V-V attn) ───┐
   query img  ──┤                                       ├──> q_feats, q_patch_feats[4 层]
                └───────────────────────────────────────┘
                ┌─── encode_image (frozen, V-V attn) ───┐
   prompt imgs ─┤      (only used when k_shots > 0)     ├──> p_feats, p_patch_feats[4 层]
                └───────────────────────────────────────┘

   static_text_features  ──┐
                           ├──> VisualAdapter   ──> (g_vl_logit, l_vl_map)
   q_feats, q_patch_feats──┘

   learnable ctx ──> encode_text_learn ──> learned_text_features ──┐
                                                                    ├──> TextualAdapter  ──> (g_tl_logit, l_tl_map)
                                          q_feats, q_patch_feats ──┘

   q_feats, q_patch_feats ─┐
   p_feats, p_patch_feats ─┴──> PQAdapter (per-layer NN align + diff + small U-Net)
                                          ──> (g_pq_logit[4], l_pq_map[4], align_map[4])

                ┌───── fusion (avg / harmonic) ─────┐
   l_vl, l_tl, l_pq, align ─┤                       ├──> pixel anomaly map
                            │ + gaussian smoothing  │
                            └───────────────────────┘
                ┌───── fusion (avg / harmonic) ─────┐
   g_vl, g_tl, g_pq, max(pixel) ─┤                  ├──> image anomaly score
                            └───────────────────────┘
```

---

## 9. 自己动手：怎么扩展或替换 Adapter

如果你要改这个项目，最常见的需求和对应入口是：

| 需求 | 改哪 |
|---|---|
| 换 prompt 模板（更专业的医学 / 工业话术） | `TextualAdapter.static_normal_list` / `static_anomaly_list` / `template_list` (`adaptclip.py:866-905`) |
| 调 CoOp 上下文长度 | `--n_ctx`（默认 12），最终影响 `ctx_vectors_pos` 形状（`adaptclip.py:913`） |
| 换 ResMLP 容量 | `--vl_reduction`（默认 4，越小越大） |
| 换 PQAdapter 卷积头宽度 | `--pq_mid_dim`（默认 128） |
| 不要 query 上下文，只看差 | 不传 `--pq_context` |
| 用更多 / 更少层 patch | `--features_list 6 12 18 24` 改为别的，注意 `pq_learner` 是按 `len(features_list)` 建 ModuleList，要重训 |
| 换融合方式 | `--fusion_type harmonic_mean / average_mean`（`test.py:381`） |
| 加第四个 adapter | 仿照 `VisualAdapter`，写个 `nn.Module`，在 `train.py` 里加进 optimizer 和 forward，在 `test.py` 里加进 fusion |

需要注意的一致性：

- 新加的 adapter 输出最好对齐到 `(B, 2, H, W)` 像素图 + `(B, 2)` 图级 logits 这两个统一接口，否则 fusion 那段要改。
- 训练脚本里 `with torch.no_grad():` 的范围只覆盖 `model.encode_image / encode_text` 这两步，新 adapter 不要在 no_grad 块里建。
- 主干的 `model.eval()` 不要去掉——LayerNorm / Dropout 都是按 eval 状态在跑，改了会影响特征数值。

---

## 10. 一句话总结

> AdaptCLIP = **冻结 CLIP（带 V-V 注意力改造）** + **TextualAdapter（CoOp 风格软 prompt）** + **VisualAdapter（双 ResMLP 微调视觉特征）** + **PQAdapter（few-shot 正常样本 NN 对齐 + 小 U-Net 分类头）** + **多分支调和/算术融合**。

三个 adapter 各司其职，参数量都很小，主干不动；这正是这套方法能在多个数据集间迁移、在零样本和小样本两种 setting 下都能跑得不错的根本原因。
