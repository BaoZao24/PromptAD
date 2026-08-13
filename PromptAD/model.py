"""PromptAD 的核心模型文件。

> 通俗导读：见 `docs/archive/model_walkthrough.md`，第一次读这份文件强烈建议先看那份。

文件章节地图（按行号粗略给出，最好结合代码当前实际位置看）：

  §1  约   1– 36 : 顶部导入 + CLIP 归一化常量
  §2  约  38–1304: 一堆 `*Channels` 类。每个类把"灰度频谱图"按一种规则
                   变成 3 通道张量，给 CLIP 当 RGB 输入用。主方案是
                   `MorphFusionGrayResidualChannels`。
  §3  约 1306–1538: 可学习的轻量小模块——CLIP 主干冻结时只训练它们。
                   ResidualVisualAdapter / VisualClassPromptAdapter /
                   LearnableScoreFusionHead / DenseMaskHead /
                   TinyMambaFusionBlock + TinyCNNPatchEncoder +
                   CNNMambaLocalBranch。每个模块最后一层都置零，初始
                   输出为 0，相当于"训练初期不影响 CLIP"。
  §4  约 1541– …: 核心两个类。
                   - PromptLearner: 生成"正常 / 异常"两类 prompt embedding。
                     里面的 normal_ctx 和 abnormal_ctx 是这份代码真正训练
                     的两组可学习向量。
                   - PromptAD:     把 CLIP + PromptLearner + 各种 gallery
                     缓存装在一起；forward/score_cached 出最终异常分数。

两条打分主腿（在 PromptAD 里）：
  文本腿 → calculate_textual_anomaly_score : 图像特征和文本原型比相似度
  视觉腿 → calculate_visual_anomaly_score  : 每个 patch 去正常 gallery 找最近邻
"""

import torch
import random
import torch.nn as nn
import cv2
import numpy as np
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from tqdm import tqdm

from .CLIPAD import SimpleTokenizer as _Tokenizer

# 本地 tokenizer：这里仅创建对象备用，不会自动补 padding / SOS / EOS。
_tokenizer = _Tokenizer()

valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms
from torchvision.transforms import functional as TF


# CLIP 预训练时使用的图像归一化参数。
# 当前项目把频谱图转成 RGB 后送入 CLIP，因此沿用 CLIP 的 mean/std。
mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]
spectrogram_mean_train = [0.5, 0.5, 0.5]
spectrogram_std_train = [0.5, 0.5, 0.5]


def _convert_to_rgb(image):
    # CLIP 的视觉编码器需要 3 通道 RGB 输入。
    return image.convert('RGB')


class Gray3Channels:
    """Pure grayscale spectrogram repeated as 3 channels.

    Corresponds to ``input_mode=gray3``. This is the conservative black/white
    input baseline: no false color, no morphology enhancement, no residual map.
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        return torch.cat([gray, gray, gray], dim=0)


class GrayLocal2DEdgeChannels:
    """Gray + local 2D residual + edge gradient.

    Corresponds to ``input_mode=gray_local2d_edge``.
    C1 keeps the raw grayscale spectrogram, C2 highlights local deviation from
    a 2D median background, and C3 highlights abrupt edges or chirp-like slopes.
    """

    def __init__(self, win=15):
        self.win = win
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _minmax_np(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)

        local_median = cv2.medianBlur(gray_u8, self.win).astype(np.float32) / 255.0
        local_residual = self._minmax_np(np.abs(gray_np - local_median)).astype(np.float32)

        grad_x = cv2.Sobel(gray_np, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_np, cv2.CV_32F, 0, 1, ksize=3)
        edge_grad = self._minmax_np(np.sqrt(grad_x * grad_x + grad_y * grad_y)).astype(np.float32)

        return torch.cat([
            gray,
            torch.from_numpy(local_residual).unsqueeze(0),
            torch.from_numpy(edge_grad).unsqueeze(0),
        ], dim=0)


class MorphFusionGrayResidualChannels:
    """🌟 当前主方案：CLAHE 对比度增强灰度 + 弱残差(alpha=0.1) + 原始灰度。

    对应 ``input_mode=morph_fusion_gray_residual_a01``。
    思路：CLAHE 把暗弱信号拉亮，残差凸显偏离背景的位置，再补一通道原始灰度
    保住未失真的能量信息。
    """

    def __init__(self, alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)

        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        gray_contrast = torch.from_numpy(gray_contrast).unsqueeze(0)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return torch.cat([
            gray_contrast,
            weak_residual,
            gray,
        ], dim=0)


class MorphFusionClaheGrayChannels:
    """新变体：CLAHE + gray + gray，去掉残差通道。

    对应 ``input_mode=morph_fusion_clahe_gray``。
    设计意图：测试"去掉残差通道"是否伤害判别能力。如果掉得少，说明 weak_residual
    本来就帮不上忙；如果掉得多，说明 weak_residual 的贡献被低估。
    通道布局：
      C1: CLAHE(gray)
      C2: gray (原本是 weak_residual)
      C3: gray
    """

    def __init__(self, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)
        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        gray_contrast = torch.from_numpy(gray_contrast).unsqueeze(0)
        return torch.cat([gray_contrast, gray, gray], dim=0)


class MorphFusionLocal2DResidualChannels:
    """🌟 新方案：CLAHE 对比度增强灰度 + 局部 2D 残差 + 原始灰度。

    对应 ``input_mode=morph_fusion_local2d_residual_a01``。
    与旧 morph_fusion_gray_residual_a01 的区别：把 weak_residual (按频率行的
    时间方向中位数残差) 换成 local 2D residual (2D 邻域中位数残差)。

    原因：weak_residual 假设异常只在时间方向偏离，对 chirp 跨频扫频和 pulse
    宽频突发都不友好；local 2D residual 用真正的二维局部邻域，对 burst/chirp/
    pulse 三种异常都能提取出清晰的偏离结构。

    通道设计：
      C1: CLAHE(gray) 对比度增强 (chirp/burst 友好)
      C2: |gray - median_filter_2d(gray, win)| min-max 归一化 (任意 2D 异常均凸显)
      C3: gray 原始灰度 (保真兜底)
    """

    def __init__(self, win=15, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.win = win
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )

    @staticmethod
    def _minmax_np(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))      # (1, H, W) in [0,1]
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)

        # C1: CLAHE 增强
        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        gray_contrast = torch.from_numpy(gray_contrast).unsqueeze(0)

        # C2: 2D 局部中位数残差
        local_median = median_filter(gray_np, size=self.win)
        residual = np.abs(gray_np - local_median)
        residual = self._minmax_np(residual).astype(np.float32)
        local_2d_residual = torch.from_numpy(residual).unsqueeze(0)

        # C3: 原始灰度
        return torch.cat([
            gray_contrast,
            local_2d_residual,
            gray,
        ], dim=0)


class MorphFusionTopHatChannels:
    """CLAHE 对比度增强灰度 + top-hat 局部显著性 + 原始灰度。

    对应 ``input_mode=morph_fusion_tophat_a01``。
    中间通道使用 white/black top-hat 的最大响应，目标是突出局部亮/暗小结构，
    同时比 local std / local residual 更少放大整片背景纹理。
    """

    def __init__(self, kernel_size=13, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.kernel_size = kernel_size
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    @staticmethod
    def _minmax_np(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)

        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        gray_contrast = torch.from_numpy(gray_contrast).unsqueeze(0)

        white_hat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, self.kernel).astype(np.float32)
        black_hat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, self.kernel).astype(np.float32)
        top_hat = self._minmax_np(np.maximum(white_hat, black_hat)).astype(np.float32)
        top_hat = torch.from_numpy(top_hat).unsqueeze(0)

        return torch.cat([
            gray_contrast,
            top_hat,
            gray,
        ], dim=0)


class MorphFusionMultiScaleResidualChannels:
    """CLAHE 对比度增强灰度 + 多尺度背景残差 + 原始灰度。

    对应 ``input_mode=morph_fusion_multiscale_residual_a01``。
    中间通道用多个 Gaussian 平滑尺度的背景差分取最大值，避免旧 row residual
    只偏向某一种信号形态。
    """

    def __init__(self, kernel_sizes=(7, 15, 31), clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.kernel_sizes = kernel_sizes
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )

    @staticmethod
    def _minmax_np(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)

        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        gray_contrast = torch.from_numpy(gray_contrast).unsqueeze(0)

        residuals = []
        for kernel_size in self.kernel_sizes:
            background = cv2.GaussianBlur(gray_np, (kernel_size, kernel_size), 0)
            residuals.append(np.abs(gray_np - background))
        residual = self._minmax_np(np.maximum.reduce(residuals)).astype(np.float32)
        residual = torch.from_numpy(residual).unsqueeze(0)

        return torch.cat([
            gray_contrast,
            residual,
            gray,
        ], dim=0)


class MorphFusionGrayResidualNoContrastChannels:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return torch.cat([
            gray,
            weak_residual,
            gray,
        ], dim=0)


class ResidualVisualAdapter(nn.Module):
    """残差式视觉特征 adapter（对应 --visual-adapter）。

    设计目的：
    冻结 CLIP 主干以后，仅训练一个轻量瓶颈 MLP，把视觉特征加上一个小幅度的残差修正：
        adapted = feature + alpha * MLP(feature)

    最后一层权重和 bias 初始化为 0，因此训练初期 adapter 输出近似于恒等映射，
    保证早期不会破坏 CLIP 自带的视觉表征，再随梯度逐步学习与频谱任务对齐的修正项。
    """

    def __init__(self, dim, bottleneck_ratio=0.25, alpha=0.2):
        super().__init__()
        # 瓶颈维度：dim * bottleneck_ratio，控制 adapter 容量，默认 1/4。
        hidden_dim = max(1, int(dim * bottleneck_ratio))
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # 把最后一层置零 → adapter 初始为零映射，等价于不影响原 CLIP 特征。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature):
        # CLIP 默认 fp16，但 adapter 为了数值稳定使用 fp32 运算，最后再转回原 dtype。
        original_dtype = feature.dtype
        adapted = feature.float() + self.alpha * self.net(feature.float())
        return adapted.to(original_dtype)


class VisualClassPromptAdapter(nn.Module):
    """VCPA：把正常视觉原型映射成软 class prompt token（对应 --visual-class-prompt）。

    PromptAD 原始 prompt 中的“class name”是固定的英文单词（例如 "Playground_spectrum"），
    它对 RF 频谱任务并不天然合适。VCPA 的做法是：
        正常视觉原型 (visual_dim) -> bottleneck MLP -> token_num × text_dim 的软 token
    然后把这些软 token 拼接到 prompt 的 class name 位置，让“类别词”也能被视觉信号驱动。

    最后一层权重置零，初始时输出全 0，软 token 不会扰动现有 prompt，再逐渐学习。
    """

    def __init__(self, visual_dim, text_dim, token_num=2, bottleneck_ratio=0.25, alpha=0.2):
        super().__init__()
        self.token_num = int(token_num)
        self.text_dim = int(text_dim)
        self.alpha = float(alpha)
        hidden_dim = max(1, int(visual_dim * bottleneck_ratio))
        # LayerNorm 让正常原型的尺度稳定，再走瓶颈 MLP 投到文本 embedding 维度。
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_num * self.text_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prototype, dtype=None):
        # prototype 形状 (P, visual_dim)；P 为正常原型数（默认 1，diverse 模式下 >1）。
        if prototype.ndim == 1:
            prototype = prototype.unsqueeze(0)
        # 投影输出 reshape 成 (P, token_num, text_dim)，作为 prompt 的软 class token。
        tokens = self.net(prototype.float()).reshape(prototype.shape[0], self.token_num, self.text_dim)
        # alpha 控制 VCPA 注入到 prompt 的强度。
        tokens = self.alpha * tokens
        if dtype is not None:
            tokens = tokens.to(dtype)
        return tokens


class LearnableScoreFusionHead(nn.Module):
    """轻量可学习融合头（对应 --learnable-score-fusion）。

    输入是 4 个标量统计量（textual 分数 + visual top-k mean / max / gap），
    输出一个对 textual 分数的残差 logit 修正：
        final_logit = logit(textual) + alpha * Head(features)

    最后一层置零 → 训练初期不改变 textual 分数；之后 BCE 监督使头部学到把
    textual 不显著但视觉显著的图像往“异常”侧推。
    """

    def __init__(self, input_dim=4, hidden_dim=4):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class DenseMaskHead(nn.Module):
    """Small pixel-supervised head over frozen CLIP patch tokens.

    This is the first implementation step for RF cross-library dense mask
    supervision. It keeps the existing PromptAD branches intact and learns two
    patch-level anomaly maps from the two CLIP patch feature streams.
    """

    def __init__(self, dim1, dim2, hidden_ratio=0.25):
        super().__init__()

        def make_head(dim):
            hidden_dim = max(32, int(dim * hidden_ratio))
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        self.raw_head = make_head(dim1)
        self.post_head = make_head(dim2)

    def forward(self, feature_map1, feature_map2, grid_size):
        b, _, _ = feature_map1.shape
        h, w = grid_size
        head_dtype = self.raw_head[0].weight.dtype
        feature_map1 = feature_map1.to(dtype=head_dtype)
        feature_map2 = feature_map2.to(dtype=head_dtype)
        raw_logits = self.raw_head(feature_map1).transpose(1, 2).reshape(b, 1, h, w)
        post_logits = self.post_head(feature_map2).transpose(1, 2).reshape(b, 1, h, w)
        return raw_logits.float(), post_logits.float()


class TextAlignedDenseProjectionHead(nn.Module):
    """APRIL-GAN style visual-to-text projection over CLIP patch tokens.

    DenseMaskHead learns a free binary map. This head instead projects patch
    features into CLIP text space and scores each patch by its abnormal-vs-normal
    text margin. The image score is later obtained by top-k pooling this map.
    """

    def __init__(self, dim1, dim2, text_dim):
        super().__init__()
        self.raw_proj = nn.Sequential(nn.LayerNorm(dim1), nn.Linear(dim1, text_dim))
        self.post_proj = nn.Sequential(nn.LayerNorm(dim2), nn.Linear(dim2, text_dim))

    @staticmethod
    def _margin_logits(projected_tokens, text_features, logit_scale):
        projected_tokens = F.normalize(projected_tokens.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        logits = logit_scale.float() * projected_tokens @ text_features.T
        normal_logit = logits[..., 0:1]
        if logits.shape[-1] == 2:
            return logits[..., 1:2] - normal_logit
        return (logits[..., 1:] - normal_logit).max(dim=-1, keepdim=True).values

    def forward(self, feature_map1, feature_map2, text_features, grid_size, logit_scale):
        b, _, _ = feature_map1.shape
        h, w = grid_size
        head_dtype = self.raw_proj[0].weight.dtype
        raw_tokens = self.raw_proj(feature_map1.to(dtype=head_dtype))
        post_tokens = self.post_proj(feature_map2.to(dtype=head_dtype))
        raw_logits = self._margin_logits(raw_tokens, text_features, logit_scale)
        post_logits = self._margin_logits(post_tokens, text_features, logit_scale)
        raw_logits = raw_logits.transpose(1, 2).reshape(b, 1, h, w)
        post_logits = post_logits.transpose(1, 2).reshape(b, 1, h, w)
        return raw_logits.float(), post_logits.float()


class TinyMambaFusionBlock(nn.Module):
    """Mamba 风格的轻量 token mixer（CNN-Mamba 局部分支组件）。

    本地实现的极简 Mamba 风：
        x -> LayerNorm -> 双路投影 (u, v)
        u 走 depthwise conv1d 在 token 维度做长程上下文聚合
        融合：u * sigmoid(v) （门控）
        out_proj -> 残差相加

    out_proj 权重置零，整块初始时近似恒等映射，叠多层不会破坏前面 CNN encoder 的特征。
    """

    def __init__(self, dim, expansion=2, kernel_size=3, dropout=0.0):
        super().__init__()
        inner_dim = max(1, int(dim * expansion))
        kernel_size = max(1, int(kernel_size))
        # depthwise conv1d 需要奇数 kernel 才能保证 same-padding。
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(dim)
        # 把 dim 投到 2 * inner_dim，再 chunk 成 (u, v)，u 走卷积、v 做门控。
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        # depthwise：每个通道一个独立 1D conv，参数量 = inner_dim * kernel_size。
        self.dwconv = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=kernel_size // 2, groups=inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        u, v = self.in_proj(x).chunk(2, dim=-1)
        # Conv1d 期望 (B, C, L)，所以先转置再做完卷积转回 (B, L, C)。
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.gelu(u)
        # u 提供局部上下文；sigmoid(v) 提供门控权重。
        x = self.out_proj(self.dropout(u * torch.sigmoid(v)))
        return residual + x


class TinyCNNPatchEncoder(nn.Module):
    """3 层 CNN，把原始图像编码成 grid_size × grid_size 的 patch token 序列。

    最后通过 AdaptiveAvgPool2d 强制下采样到目标网格大小，
    保证输出 token 数和 ViT 主干的 patch 网格一致，可以直接对齐分数图。
    """

    def __init__(self, dim, grid_size):
        super().__init__()
        hidden1 = max(32, dim // 2)
        hidden2 = max(32, dim)
        # stem：两次下采样 + 三次卷积，输出通道数与 ViT 特征维度对齐。
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden1, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )
        # 自适应池化到目标 grid，使不同输入分辨率最终都得到同样数量的 token。
        self.pool = nn.AdaptiveAvgPool2d(grid_size)

    def forward(self, image):
        feat = self.stem(image.float())
        # (B, C, H, W) -> (B, H*W, C)，与 transformer token 序列布局一致。
        tokens = self.pool(feat).flatten(2).transpose(1, 2)
        return tokens


class CNNMambaLocalBranch(nn.Module):
    """CNN + ViT + Mamba 混合的局部异常分支（对应 --cnn-vit-mamba-fusion）。

    流程：image → TinyCNNPatchEncoder → 多层 TinyMambaFusionBlock → LayerNorm → L2 归一化。

    输出 token 用于：
    1. 与正常样本的 cnn_mamba_gallery 做最近邻距离 → 局部异常 patch map / image score；
    2. 训练阶段聚合后与 CLIP-ViT 全局特征做对齐损失（cnn_mamba_align_lambda）。
    """

    def __init__(self, dim, grid_size, num_blocks=1, dropout=0.0):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.cnn_encoder = TinyCNNPatchEncoder(dim=dim, grid_size=self.grid_size)
        # 多层 Mamba block 串联，扩大 token 之间的有效感受野。
        self.mamba_blocks = nn.ModuleList([
            TinyMambaFusionBlock(dim=dim, dropout=dropout)
            for _ in range(max(1, int(num_blocks)))
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, image):
        tokens = self.cnn_encoder(image)
        for block in self.mamba_blocks:
            tokens = block(tokens)
        # 输出统一 L2 归一化，之后与 gallery 做余弦相似度等价于内积。
        return F.normalize(self.out_norm(tokens), dim=-1)


class PromptLearner(nn.Module):
    # PromptLearner 负责生成三类文本 prompt：
    # 1. normal_prompts：正常样本 prompt，包含可学习 normal_ctx；
    # 2. abnormal_prompts_handle：手工异常模板 prompt，来自 ad_prompts.py；
    # 3. abnormal_prompts_learned：可学习异常 prompt，包含可学习 abnormal_ctx。
    #
    # 这里的“可学习 prompt”不是普通英文单词，而是 CLIP 文本 embedding 空间中的可训练向量。
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre,
                 dataset_name=None, prompt_mode="rf", text_prototype_mode="single",
                 visual_class_prompt=False, visual_class_token_num=2,
                 visual_class_prompt_bottleneck_ratio=0.25, visual_class_prompt_alpha=0.2):
        super().__init__()

        # 根据 CLIP 精度选择 prompt 向量的数据类型。
        # fp16 可以节省显存，float32 更稳定但更耗显存。
        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        # rf 模式会把 RF 任务中的信号类型解析成更适合 CLIP 的自然语言。
        # legacy 模式保留旧逻辑，便于做 ablation。
        use_grouped_abnormal = (
            text_prototype_mode in {"grouped_max", "grouped_mean", "grouped_meanmax", "grouped_softmax"}
            and is_rf_prompt_class(classname, dataset_name)
        )
        state_anomaly1 = get_abnormal_prompt_states(
            classname, dataset_name, prompt_mode
        )
        classname = get_prompt_classname(
            classname, dataset_name, prompt_mode
        )
        grouped_abnormal_specs = get_grouped_rf_abnormal_prompt_specs() if use_grouped_abnormal else []

        # CLIP 文本 token 的 embedding 维度。
        # 后续可学习 prompt 向量必须和 CLIP token embedding 维度一致。
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 随机初始化可学习 prompt 向量。
        # normal_ctx_vectors: 正常 prompt 中可学习的上下文 token。
        # abnormal_ctx_vectors: 异常 prompt 中额外可学习的异常 token。
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)

        # 使用小方差正态分布初始化，避免一开始扰动 CLIP 文本空间过大。
        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        # 这里的 N / A 只是占位符，用于构造可 tokenize 的文本模板。
        # 真正送入 CLIP text encoder 时，这些位置会被 normal_ctx / abnormal_ctx 向量替换。
        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        # nn.Parameter 表示这些向量会被 optimizer 更新。
        # 当前训练脚本主要优化 model.prompt_learner.parameters()，
        # 因此 normal_ctx 和 abnormal_ctx 都会参与训练。
        self.visual_class_prompt = visual_class_prompt
        self.visual_class_token_num = int(visual_class_token_num)
        self.visual_class_prompt_bottleneck_ratio = visual_class_prompt_bottleneck_ratio
        self.visual_class_prompt_alpha = visual_class_prompt_alpha
        self.visual_class_prompt_adapter = None
        self.visual_class_prompt_prototype = None

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)

        # 正常 prompt 模板，例如：
        # "N N N N radio frequency spectrum."
        # 其中 N 的位置后续会被 normal_ctx 替换成可学习向量。
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # 手工异常 prompt，例如：
        # "N N N N abnormal radio frequency spectrum."
        # "N N N N radio frequency spectrum with defect."
        # 注意：它仍然使用 normal_ctx 作为前缀上下文，但异常语义来自手工模板。
        abnormal_prompts_handle = []
        abnormal_handle_group_slices = []
        if use_grouped_abnormal:
            for group_name, group_classname, group_states in grouped_abnormal_specs:
                start = len(abnormal_prompts_handle)
                abnormal_prompts_handle.extend(
                    normal_prompt_prefix + " " + state.format(group_classname) + "."
                    for state in group_states
                    for _ in range(n_pro)
                )
                abnormal_handle_group_slices.append((group_name, start, len(abnormal_prompts_handle)))
        else:
            abnormal_prompts_handle = [
                normal_prompt_prefix + " " + state.format(classname) + "."
                for state in state_anomaly1
                for _ in range(n_pro)
            ]

        # 手工异常 prompt 数量。
        # 每个手工模板会和 n_pro 组 normal_ctx 组合。
        self.n_ab_handle = len(abnormal_prompts_handle) // n_pro
        self.use_grouped_abnormal = use_grouped_abnormal
        self.abnormal_handle_group_slices = abnormal_handle_group_slices

        # 可学习异常 prompt，例如：
        # "N N N N A A A A radio frequency spectrum."
        # 其中 N 位置使用 normal_ctx，A 位置使用 abnormal_ctx。
        # abnormal_ctx 是该项目中真正“学习异常语义”的核心参数。
        abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + classname + "." for _ in range(n_pro_ab) for _ in range(n_pro)]
        self.normal_prompt_texts = normal_prompts
        self.abnormal_prompt_texts_handle = abnormal_prompts_handle
        self.abnormal_prompt_texts_learned = abnormal_prompts_learned

        # abnormal_prompts = abnormal_prompts_learned + abnormal_prompts_handle

        # tokenize 后得到的是 token id，不是 embedding。
        # 后面需要先通过 CLIP token_embedding 得到每个 token 的向量。
        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        tokenized_abnormal_prompts_handle = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_handle])
        tokenized_abnormal_prompts_learned = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_learned])

        # 这里不需要训练 CLIP 原始 token embedding，因此 no_grad。
        # 之后会从这些 embedding 中取出固定的 SOS、类别名、EOS 等 token，
        # 只把中间的可学习 token 位置替换成 normal_ctx / abnormal_ctx。
        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_handle = clip_model.token_embedding(tokenized_abnormal_prompts_handle).type(dtype)
            abnormal_embedding_learned = clip_model.token_embedding(tokenized_abnormal_prompts_learned).type(dtype)

        # register_buffer 表示这些张量属于模型状态，但不是可训练参数。
        # 它们会跟随模型移动到 CPU/GPU，也会出现在 state_dict 中。
        #
        # prefix 通常是 SOS token；
        # suffix 包含类别名、标点、EOS 等固定 token。
        # 中间的上下文 token 会在 forward 中被可学习向量替换。
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])

        # 手工异常 prompt：只替换 normal_ctx 部分，后面的异常模板文本保持固定。
        self.register_buffer("abnormal_token_prefix_handle", abnormal_embedding_handle[:, :1, :])
        self.register_buffer("abnormal_token_suffix_handle", abnormal_embedding_handle[:, 1 + n_ctx:, :])

        # 可学习异常 prompt：同时替换 normal_ctx 和 abnormal_ctx。
        # 因此 suffix 要从 1 + n_ctx + n_ctx_ab 之后开始保留。
        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])

        # 保存 prompt 相关超参数和 tokenized 结果。
        # encode_text_embeddings 需要原始 token id 来确定文本结束位置等信息。
        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned
        # self.tokenized_abnormal_prompts = torch.cat([tokenized_abnormal_prompts_handle, tokenized_abnormal_prompts_learned], dim=0)
        # self.tokenized_abnormal_prompts = tokenized_abnormal_prompts_handle
        # self.name_lens = name_lens

    def set_visual_class_prompt_adapter(self, adapter):
        object.__setattr__(self, 'visual_class_prompt_adapter', adapter)

    def set_visual_class_prompt_prototype(self, prototype):
        self.visual_class_prompt_prototype = prototype

    def _get_visual_class_prompt_tokens(self, suffix):
        if (
            not self.visual_class_prompt
            or self.visual_class_prompt_adapter is None
            or self.visual_class_prompt_prototype is None
        ):
            return None
        prototype = self.visual_class_prompt_prototype.to(device=suffix.device)
        tokens = self.visual_class_prompt_adapter(prototype, dtype=suffix.dtype)
        if tokens.shape[0] > 1:
            tokens = tokens.mean(dim=0, keepdim=True)
        return tokens

    def _apply_visual_class_prompt(self, suffix, repeat_count=1):
        tokens = self._get_visual_class_prompt_tokens(suffix)
        if tokens is None:
            return suffix
        tokens = tokens.expand(suffix.shape[0] // repeat_count, -1, -1)
        if repeat_count > 1:
            tokens = tokens.unsqueeze(0).expand(repeat_count, -1, -1, -1).reshape(-1, tokens.shape[1], tokens.shape[2])
        token_num = min(tokens.shape[1], suffix.shape[1])
        suffix = suffix.clone()
        suffix[:, :token_num, :] = suffix[:, :token_num, :] + tokens[:, :token_num, :]
        return suffix

    def forward(self):

        # 生成正常 prompt 的 embedding 序列。
        # 结构为：[SOS] + normal_ctx + [classname + "." + EOS]
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self._apply_visual_class_prompt(self.normal_token_suffix)

        normal_prompts = torch.cat(
            [
                normal_prefix,  # (n_pro, 1, dim)
                normal_ctx,     # (n_pro, n_ctx, dim)
                normal_suffix,  # (n_pro, *, dim)
            ],
            dim=1,
        )

        # 生成手工异常 prompt 的 embedding 序列。
        # 结构为：[SOS] + normal_ctx + [手工异常模板 + classname + "." + EOS]
        # 这里没有使用 abnormal_ctx，异常语义主要来自人工写的 abnormal template。
        n_ab_handle = self.n_ab_handle

        n_pro, n_ctx, dim = normal_ctx.shape
        # 每个手工异常模板都要配上 n_pro 组 normal_ctx，
        # 因此需要把 normal_ctx 复制 n_ab_handle 份。
        normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_handle, -1, -1, -1).reshape(-1, n_ctx, dim)

        abnormal_prefix_handle = self.abnormal_token_prefix_handle
        abnormal_suffix_handle = self._apply_visual_class_prompt(
            self.abnormal_token_suffix_handle,
            repeat_count=n_ab_handle,
        )

        abnormal_prompts_handle = torch.cat(
            [
                abnormal_prefix_handle,     # (n_pro * n_ab_handle, 1, dim)
                normal_ctx1,                # (n_pro * n_ab_handle, n_ctx, dim)
                abnormal_suffix_handle,     # (n_pro * n_ab_handle, *, dim)
            ],
            dim=1,
        )

        # 生成可学习异常 prompt 的 embedding 序列。
        # 结构为：[SOS] + normal_ctx + abnormal_ctx + [classname + "." + EOS]
        # abnormal_ctx 会通过训练自动学习“异常”的方向。
        abnormal_prefix_learned = self.abnormal_token_prefix_learned
        abnormal_suffix_learned = self._apply_visual_class_prompt(
            self.abnormal_token_suffix_learned,
            repeat_count=self.n_pro_ab,
        )
        abnormal_ctx = self.abnormal_ctx
        n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape

        # normal_ctx2: 为每一组 abnormal_ctx 复制 normal_ctx。
        normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
        # abnormal_ctx: 为每一组 normal_ctx 复制 abnormal_ctx。
        abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

        abnormal_prompts_learned = torch.cat(
            [
                abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
            ],
            dim=1,
        )

        # abnormal_prompts = torch.cat([abnormal_prompts_handle, abnormal_prompts_learned], dim=0)
        # abnormal_prompts = abnormal_prompts_handle

        # 返回的是“prompt embedding 序列”，不是字符串。
        # 后续会送入 CLIP 的 encode_text_embeddings 得到文本特征。
        return normal_prompts, abnormal_prompts_handle, abnormal_prompts_learned


class PromptAD(torch.nn.Module):
    # PromptAD 是整个异常检测模型的主类。
    # 它同时包含：
    # 1. CLIP 图像编码器：把频谱图转成视觉特征；
    # 2. CLIP 文本编码器：把 prompt embedding 转成文本特征；
    # 3. PromptLearner：生成可学习的正常/异常 prompt；
    # 4. feature_gallery1/2：保存正常训练样本的 patch 特征，用于视觉异常分数；
    # 5. text_features：保存正常/异常文本特征，用于文本异常分数。
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name,  precision='fp16', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(PromptAD, self).__init__()

        # k-shot 中的 shot 数。
        self.shot = kwargs['k_shot']

        # 输出异常图的目标尺寸。
        # 后续 anomaly_map 会被插值到这个大小，用于可视化和指标计算。
        self.out_size_h = out_size_h
        self.out_size_w = out_size_w

        # 当前代码强制使用 fp16。
        # 好处是节省显存；代价是数值精度略低。
        self.precision = 'fp16'

        self.device = device

        # 加载 CLIP 主体、创建 PromptLearner、初始化文本/视觉特征缓存。
        self.prompt_mode = kwargs.get('prompt_mode', 'rf')
        self.text_prototype_mode = kwargs.get('text_prototype_mode', 'single')
        self.dataset_name = kwargs.get('dataset', None)
        self.input_mode = self._resolve_input_mode(kwargs.get('input_mode', 'auto'), self.dataset_name)
        self.cls_score_mode = kwargs.get('cls_score_mode', 'text_only')
        self.visual_topk_ratio = kwargs.get('visual_topk_ratio', 0.05)
        self.visual_gallery_chunk_size = int(kwargs.get('visual_gallery_chunk_size', 1024))
        self.visual_score_alpha = kwargs.get('visual_score_alpha', 1.0)
        self.visual_score_beta = kwargs.get('visual_score_beta', 1.0)
        self.visual_score_gamma = kwargs.get('visual_score_gamma', 0.0)
        self.visual_freq_position_weight = kwargs.get('visual_freq_position_weight', 0.0)
        self.normal_dist_ridge = kwargs.get('normal_dist_ridge', 1e-4)
        self.use_learnable_score_fusion = kwargs.get('learnable_score_fusion', False)
        self.use_dense_mask_branch = kwargs.get('dense_mask_branch', False)
        self.dense_mask_hidden_ratio = kwargs.get('dense_mask_hidden_ratio', 0.25)
        self.dense_mask_score_alpha = kwargs.get('dense_mask_score_alpha', 1.0)
        self.use_text_aligned_dense = kwargs.get('text_aligned_dense', False)
        self.text_aligned_dense_score_beta = kwargs.get('text_aligned_dense_score_beta', 1.0)
        self.ta_normal_quantile = float(kwargs.get('ta_normal_quantile', 0.95))
        self.ta_normal_excess_scale_floor = float(kwargs.get('ta_normal_excess_scale_floor', 1e-6))
        self.ta_normal_topk_threshold = None
        self.ta_normal_topk_scale = None
        self.ta_normal_max_threshold = None
        self.ta_normal_max_scale = None
        self.use_cnn_vit_mamba_fusion = kwargs.get('cnn_vit_mamba_fusion', False)
        self.use_rn50_visual_fusion = kwargs.get('rn50_visual_fusion', False)
        self.rn50_pretrained = kwargs.get('rn50_pretrained', 'openai')
        self.rn50_visual_beta = kwargs.get('rn50_visual_beta', 0.05)
        self.rn50_score_mode = kwargs.get('rn50_score_mode', 'global')
        self.rn50_fusion_mode = kwargs.get('rn50_fusion_mode', 'score')
        self.rn50_local_topk_ratio = kwargs.get('rn50_local_topk_ratio', 0.1)
        self.rn50_guidance_temperature = kwargs.get('rn50_guidance_temperature', 0.2)
        self.cnn_mamba_beta = kwargs.get('cnn_mamba_beta', None)
        if self.cnn_mamba_beta is None:
            self.cnn_mamba_beta = kwargs.get('cnn_vit_mamba_alpha', 0.05)
        self.cnn_mamba_map_beta = kwargs.get('cnn_mamba_map_beta', None)
        if self.cnn_mamba_map_beta is None:
            self.cnn_mamba_map_beta = kwargs.get('cnn_vit_mamba_patch_alpha', 0.0)
        self.cnn_mamba_pool_topk_ratio = kwargs.get('cnn_mamba_pool_topk_ratio', 0.2)
        self.cnn_mamba_pool_temperature = kwargs.get('cnn_mamba_pool_temperature', 0.1)
        self.cnn_vit_mamba_num_blocks = kwargs.get('cnn_vit_mamba_num_blocks', 2)
        self.cnn_vit_mamba_dropout = kwargs.get('cnn_vit_mamba_dropout', 0.0)
        self.learnable_score_fusion_hidden_dim = kwargs.get('learnable_score_fusion_hidden_dim', 4)
        self.learnable_score_fusion_alpha = kwargs.get('learnable_score_fusion_alpha', 0.25)
        self.use_visual_adapter = kwargs.get('visual_adapter', False)
        self.adapter_bottleneck_ratio = kwargs.get('adapter_bottleneck_ratio', 0.25)
        self.adapter_alpha = kwargs.get('adapter_alpha', 0.2)
        self.use_visual_class_prompt = kwargs.get('visual_class_prompt', False)
        self.visual_class_token_num = kwargs.get('visual_class_token_num', 2)
        self.visual_class_prompt_bottleneck_ratio = kwargs.get('visual_class_prompt_bottleneck_ratio', 0.25)
        self.visual_class_prompt_alpha = kwargs.get('visual_class_prompt_alpha', 0.2)
        self.visual_class_prototype_mode = kwargs.get('visual_class_prototype_mode', 'mean')
        self.visual_class_prototype_num = kwargs.get('visual_class_prototype_num', 1)
        self.use_visual_lora = kwargs.get('visual_lora', False)
        self.visual_lora_rank = kwargs.get('visual_lora_rank', 4)
        self.visual_lora_alpha = kwargs.get('visual_lora_alpha', 8.0)
        self.visual_lora_dropout = kwargs.get('visual_lora_dropout', 0.0)
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset)
        self.phrase_form = '{}'
        self.device = device

        # 文本特征构建版本。
        # V1：先编码所有 prompt，再对正常 prompt 和异常 prompt 分别求平均。
        # V2：逐条 prompt 编码并归一化后再拼接；当前默认不用。
        self.version = 'V1'

        # 图像预处理流程：
        # 1. resize 到 img_resize；
        # 2. center crop 到 img_cropsize；
        # 3. 转成 RGB；
        # 4. 转 Tensor；
        # 5. 使用 CLIP mean/std 做归一化。
        pre_resize_crop = [
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
        ]
        if self.input_mode == 'gray3':
            self.transform = transforms.Compose(pre_resize_crop + [
                Gray3Channels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'gray_local2d_edge':
            self.transform = transforms.Compose(pre_resize_crop + [
                GrayLocal2DEdgeChannels(win=15),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_local2d_residual_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionLocal2DResidualChannels(win=15),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_tophat_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionTopHatChannels(kernel_size=13),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_multiscale_residual_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionMultiScaleResidualChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_clahe_gray':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionClaheGrayChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_no_contrast_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualNoContrastChannels(alpha=0.1),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        else:
            self.transform = transforms.Compose(pre_resize_crop + [
                _convert_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train)])

        # ground truth mask 的预处理流程。
        # mask 是离散标签图，因此 resize 时使用 NEAREST，避免插值产生非 0/1 值。
        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])

    @staticmethod
    def _resolve_input_mode(input_mode, dataset_name):
        if input_mode != 'auto':
            return input_mode
        if dataset_name in {'spectrum', 'sample', 'deceptive_signal', 'burst_signal', 'dsss_signal', 'chirp_signal', 'pulse_signal', 'wideband_pulse', 'rf_spe_png'}:
            return 'morph_fusion_gray_residual_a01'
        return 'rgb'

    def set_normal_bg_stats(self, median, mad):
        """Inject precomputed normal background statistics into channel transform."""
        if hasattr(self, '_normal_bg_transform'):
            self._normal_bg_transform.set_bg_stats(median, mad)
        else:
            raise RuntimeError(
                'set_normal_bg_stats called but current input_mode does not use '
                'normal_bg_deviation. Check input_mode argument.'
            )

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):

        # 检查 CLIP backbone 和预训练数据集是否合法。
        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        # 创建 CLIP 模型和 tokenizer。
        # 这里的 CLIPAD 是项目内封装过的 CLIP 版本，不是 openai/clip 原始接口。
        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)

        # CLIP 主体默认处于 eval 模式。
        # 当前旧分支中，主要训练 PromptLearner，不训练 CLIP 主干。
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if self.use_visual_lora and hasattr(model.visual, 'visual_lora_rank'):
            model.visual.visual_lora_rank = self.visual_lora_rank
            model.visual.visual_lora_alpha = self.visual_lora_alpha
            model.visual.visual_lora_dropout = self.visual_lora_dropout

        # 创建 prompt 学习器。
        # PromptLearner 内部包含 normal_ctx 和 abnormal_ctx 两组可训练参数。
        self.prompt_learner = PromptLearner(
            n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision,
            dataset_name=self.dataset_name,
            prompt_mode=self.prompt_mode,
            text_prototype_mode=self.text_prototype_mode,
            visual_class_prompt=self.use_visual_class_prompt,
            visual_class_token_num=self.visual_class_token_num,
            visual_class_prompt_bottleneck_ratio=self.visual_class_prompt_bottleneck_ratio,
            visual_class_prompt_alpha=self.visual_class_prompt_alpha,
        )
        self.model = model.to(self.device)
        self.rn50_model = None
        self.rn50_output_dim = 0
        self.rn50_local_dim = 1
        if self.use_rn50_visual_fusion:
            rn50_model, _, _ = CLIPAD.create_model_and_transforms(
                model_name='RN50',
                pretrained=self.rn50_pretrained,
                precision=self.precision,
            )
            rn50_model.eval()
            for p in rn50_model.parameters():
                p.requires_grad_(False)
            self.rn50_model = rn50_model.to(self.device)
            self.rn50_output_dim = self.rn50_model.visual.output_dim
            self.rn50_local_dim = self.rn50_model.visual.layer4[-1].bn3.num_features
        self.visual_adapters = None
        if self.use_visual_adapter:
            adapter_dims = {
                'output': self.model.visual.output_dim,
                'embed': self.model.visual.embed_dim,
            }
            self.visual_adapters = nn.ModuleDict()
            for name, dim in adapter_dims.items():
                if str(dim) not in self.visual_adapters:
                    self.visual_adapters[str(dim)] = ResidualVisualAdapter(
                        dim=dim,
                        bottleneck_ratio=self.adapter_bottleneck_ratio,
                        alpha=self.adapter_alpha,
                    )
            self.visual_adapters = self.visual_adapters.to(self.device)

        self.visual_class_prompt_adapter = None
        if self.use_visual_class_prompt:
            self.visual_class_prompt_adapter = VisualClassPromptAdapter(
                visual_dim=self.model.visual.output_dim,
                text_dim=self.model.ln_final.weight.shape[0],
                token_num=self.visual_class_token_num,
                bottleneck_ratio=self.visual_class_prompt_bottleneck_ratio,
                alpha=self.visual_class_prompt_alpha,
            ).to(self.device)
            self.prompt_learner.set_visual_class_prompt_adapter(self.visual_class_prompt_adapter)

        self.score_fusion_head = None
        if self.use_learnable_score_fusion:
            self.score_fusion_head = LearnableScoreFusionHead(
                input_dim=4,
                hidden_dim=self.learnable_score_fusion_hidden_dim,
            ).to(self.device)

        self.dense_mask_head = None
        if self.use_dense_mask_branch:
            self.dense_mask_head = DenseMaskHead(
                dim1=self.model.visual.embed_dim,
                dim2=self.model.visual.embed_dim,
                hidden_ratio=self.dense_mask_hidden_ratio,
            ).to(self.device)

        self.text_aligned_dense_head = None
        if self.use_text_aligned_dense:
            self.text_aligned_dense_head = TextAlignedDenseProjectionHead(
                dim1=self.model.visual.embed_dim,
                dim2=self.model.visual.embed_dim,
                text_dim=self.model.visual.output_dim,
            ).to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None

        # CLIP 视觉分支的 patch 网格大小，例如 15×15。
        # 后续 patch 级异常图会 reshape 成这个网格。
        self.grid_size = model.visual.grid_size
        self.cnn_mamba_local_branch = None
        if self.use_cnn_vit_mamba_fusion:
            self.cnn_mamba_local_branch = CNNMambaLocalBranch(
                dim=self.model.visual.output_dim,
                grid_size=self.grid_size,
                num_blocks=self.cnn_vit_mamba_num_blocks,
                dropout=self.cnn_vit_mamba_dropout,
            ).to(self.device)
        self.visual_gallery = None

        # feature_gallery1 / feature_gallery2 是视觉异常分支的正常样本特征库。
        # 训练开始时会用正常训练样本的 patch 特征覆盖这里。
        #
        # shape: (shot * grid_h * grid_w, embed_dim)
        # 例如 shot=1, grid=15×15，则每个 gallery 有 225 个正常 patch 特征。
        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        cnn_mamba_gallery = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.output_dim))
        self.register_buffer("cnn_mamba_gallery", cnn_mamba_gallery)
        cnn_mamba_global_gallery = torch.zeros((self.shot, self.model.visual.output_dim))
        self.register_buffer("cnn_mamba_global_gallery", cnn_mamba_global_gallery)

        normal_global_mean = torch.zeros((self.model.visual.output_dim,))
        normal_global_center = torch.zeros((self.model.visual.output_dim,))
        normal_global_var = torch.ones((self.model.visual.output_dim,))
        self.register_buffer("normal_global_mean", normal_global_mean)
        self.register_buffer("normal_global_center", normal_global_center)
        self.register_buffer("normal_global_var", normal_global_var)

        rn50_dim = self.rn50_output_dim if self.use_rn50_visual_fusion else self.model.visual.output_dim
        rn50_gallery = torch.zeros((self.shot, rn50_dim))
        self.register_buffer("rn50_gallery", rn50_gallery)
        rn50_local_gallery = torch.zeros((self.shot * 49, self.rn50_local_dim))
        self.register_buffer("rn50_local_gallery", rn50_local_gallery)

        # text_features 保存最终用于推理的文本原型：
        # text_features[0] = 正常文本特征；
        # single 模式下 text_features[1] = 单一异常文本特征；
        # grouped_* 模式下 text_features[1:] = burst/chirp/dsss 异常文本特征。
        if self.text_prototype_mode in {'grouped_max', 'grouped_mean', 'grouped_meanmax', 'grouped_softmax'} and self.prompt_learner.abnormal_handle_group_slices:
            num_text_features = 1 + len(self.prompt_learner.abnormal_handle_group_slices)
        else:
            num_text_features = 2
        text_features = torch.zeros((num_text_features, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)

        # 如果使用 fp16，把缓存特征也转成 half，减少显存占用。
        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.cnn_mamba_gallery = self.cnn_mamba_gallery.half()
            self.cnn_mamba_global_gallery = self.cnn_mamba_global_gallery.half()
            self.rn50_gallery = self.rn50_gallery.half()
            self.rn50_local_gallery = self.rn50_local_gallery.half()
            self.text_features  = text_features.half()

        # 保存 tokenized prompt。
        # build_text_feature_gallery 时需要这些 token id 来调用 CLIP text encoder。
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)

    def set_visual_class_prototype(self, global_features):
        # 给 VCPA 准备“正常视觉原型”。
        # 来源：所有正常训练样本的 cls_feature（已归一化）。
        # 两种模式：
        # - mean：直接平均成一个原型，等价于正常类别中心；
        # - diverse：farthest-point sampling 选 prototype_num 个互相远离的正常样本，
        #   保留场景内多样性，避免单一中心丢掉子分布信息。
        if self.visual_class_prompt_adapter is None:
            return
        normal_features = F.normalize(global_features.float(), dim=-1)
        mode = self.visual_class_prototype_mode
        prototype_num = max(1, int(self.visual_class_prototype_num))
        if mode == 'mean' or prototype_num == 1 or normal_features.shape[0] == 1:
            # 正常样本只有一张或显式要求 mean → 直接平均。
            prototype = normal_features.mean(dim=0, keepdim=True)
        elif mode == 'diverse':
            # diverse：贪心 farthest-point sampling。
            k = min(prototype_num, normal_features.shape[0])
            # 第一个原型选离全体均值最近的样本（相对“典型”的正常样本）。
            center = F.normalize(normal_features.mean(dim=0, keepdim=True), dim=-1)
            first_idx = (1.0 - normal_features @ center.t()).argmin()
            selected = [int(first_idx.item())]
            # min_dist[i] = 当前已选集合中样本到 i 的最小余弦距离。
            min_dist = 1.0 - normal_features @ normal_features[first_idx:first_idx + 1].t()
            min_dist = min_dist.squeeze(1)
            for _ in range(1, k):
                # 已选样本距离自己置 -1，避免重复选中。
                min_dist[selected] = -1.0
                # 选取离已选集合最远的样本。
                next_idx = min_dist.argmax()
                selected.append(int(next_idx.item()))
                # 更新到所有已选样本的最小距离。
                dist_to_next = 1.0 - normal_features @ normal_features[next_idx:next_idx + 1].t()
                min_dist = torch.minimum(min_dist, dist_to_next.squeeze(1))
            prototype = normal_features[selected]
        else:
            raise ValueError(f'Unsupported visual_class_prototype_mode: {mode}')
        # 把原型交给 PromptLearner，由 PromptLearner.forward() 时通过 VCPA 注入软 class token。
        self.prompt_learner.set_visual_class_prompt_prototype(prototype.detach())

    def trainable_parameters(self):
        # 收集所有需要梯度更新的参数。
        # 注意：CLIP 主干（model.visual / model.transformer）默认不在内，因此处于冻结状态；
        # 仅 PromptLearner 与各个可选 adapter 模块参与训练。
        params = list(self.prompt_learner.parameters())
        if self.visual_adapters is not None:
            params.extend(self.visual_adapters.parameters())
        if self.visual_class_prompt_adapter is not None:
            params.extend(self.visual_class_prompt_adapter.parameters())
        if self.score_fusion_head is not None:
            params.extend(self.score_fusion_head.parameters())
        if self.dense_mask_head is not None:
            params.extend(self.dense_mask_head.parameters())
        if self.text_aligned_dense_head is not None:
            params.extend(self.text_aligned_dense_head.parameters())
        if self.cnn_mamba_local_branch is not None:
            params.extend(self.cnn_mamba_local_branch.parameters())
        if self.use_visual_lora:
            # LoRA：CLIP 视觉 transformer attention 中名字含 'lora_' 的低秩矩阵。
            # 主干其它参数仍冻结。
            params.extend(
                p for name, p in self.model.visual.named_parameters()
                if 'lora_' in name and p.requires_grad
            )
        return params

    def trainable_parameter_groups(self, base_lr, prompt_lr=None, visual_adapter_lr=None,
                                   visual_class_prompt_lr=None, score_fusion_lr=None,
                                   cnn_mamba_lr=None, visual_lora_lr=None,
                                   dense_mask_lr=None, text_aligned_dense_lr=None):
        # 给优化器按模块切分参数组，便于为不同模块设置不同学习率。
        # 任意 *_lr 为 None 时，该组使用 base_lr。
        groups = []

        def add_group(params, lr):
            params = list(params)
            if params:
                groups.append({'params': params, 'lr': base_lr if lr is None else lr})

        add_group(self.prompt_learner.parameters(), prompt_lr)
        if self.visual_adapters is not None:
            add_group(self.visual_adapters.parameters(), visual_adapter_lr)
        if self.visual_class_prompt_adapter is not None:
            add_group(self.visual_class_prompt_adapter.parameters(), visual_class_prompt_lr)
        if self.score_fusion_head is not None:
            add_group(self.score_fusion_head.parameters(), score_fusion_lr)
        if self.dense_mask_head is not None:
            add_group(self.dense_mask_head.parameters(), dense_mask_lr)
        if self.text_aligned_dense_head is not None:
            add_group(self.text_aligned_dense_head.parameters(), text_aligned_dense_lr)
        if self.cnn_mamba_local_branch is not None:
            add_group(self.cnn_mamba_local_branch.parameters(), cnn_mamba_lr)
        if self.use_visual_lora:
            add_group(
                (p for name, p in self.model.visual.named_parameters() if 'lora_' in name and p.requires_grad),
                visual_lora_lr,
            )
        return groups

    def build_learnable_fusion_features(self, textual_anomaly, visual_anomaly_map):
        # 为可学习融合头构造 4 维输入特征：
        # 1) textual_anomaly：CLIP 文本分支给出的图像级异常分数；
        # 2) mean_topk：visual anomaly map 中 top-k patch 的均值（视觉显著程度）；
        # 3) max_score：visual anomaly map 的最大值（最尖锐的异常 patch 强度）；
        # 4) gap = max - mean_topk：分数“尖锐度”，越大说明只有少量 patch 被点亮。
        # 这 4 个标量信息互补：让融合头判断“文本和视觉是否一致地认为异常”。
        patch_scores = visual_anomaly_map.squeeze(1)
        n, h, w = patch_scores.shape
        flat_scores = patch_scores.reshape(n, -1)
        topk = max(1, int(flat_scores.shape[1] * self.visual_topk_ratio))
        topk_scores, _ = torch.topk(flat_scores, k=topk, dim=1)
        mean_topk = topk_scores.mean(dim=1)
        max_score = flat_scores.max(dim=1).values
        gap = max_score - mean_topk
        return torch.stack([textual_anomaly, mean_topk, max_score, gap], dim=-1)

    def calculate_learnable_fusion_logit(self, textual_anomaly, visual_anomaly_map):
        # 在 textual 分数的 logit 上做残差校正：
        #   final_logit = logit(textual_anomaly) + alpha * Head(features)
        # 用 sigmoid 还原成 [0,1] 概率作为图像级最终分数。
        if self.score_fusion_head is None:
            raise RuntimeError('calculate_learnable_fusion_logit requires --learnable-score-fusion True')
        head_device = next(self.score_fusion_head.parameters()).device
        if not torch.is_tensor(textual_anomaly):
            textual_anomaly = torch.as_tensor(textual_anomaly, device=head_device, dtype=visual_anomaly_map.dtype)
        else:
            textual_anomaly = textual_anomaly.to(device=head_device, dtype=visual_anomaly_map.dtype)
        visual_anomaly_map = visual_anomaly_map.to(head_device)
        features = self.build_learnable_fusion_features(textual_anomaly, visual_anomaly_map)
        delta = self.score_fusion_head(features).squeeze(-1)
        # logit 在 textual 端做残差，避免 textual_anomaly 已经接近 0/1 时数值溢出。
        base_logit = torch.logit(textual_anomaly.clamp(1e-4, 1.0 - 1e-4))
        return base_logit + float(self.learnable_score_fusion_alpha) * delta

    def encode_rn50_image(self, image):
        # ── RN50 视觉分支（对应 --rn50-visual-fusion） ──────────────────────────
        # 结构：冻结的 CLIP-RN50 + memory-bank 距离打分。
        # 与 ViT 主干互补：RN50 的局部归纳偏置不同，能在主干失误的样本上提供独立证据。
        #
        # 子函数分工：
        #   encode_rn50_image     : 取 RN50 全局 embedding（已 L2 归一化）
        #   encode_rn50_local     : 取 RN50 layer4 局部 token map（未做 attention pool）
        #   build_rn50_gallery    : 用全部正常样本特征构建距离 gallery
        #   calculate_rn50_global_score / local_score / guided_vit_score : 三种打分模式
        # ───────────────────────────────────────────────────────────────────
        if self.rn50_model is None:
            raise RuntimeError('encode_rn50_image requires --rn50-visual-fusion True')
        # CLIP-RN50 训练分辨率 224，做线性插值统一尺寸。
        image = F.interpolate(image.float(), size=(224, 224), mode='bilinear', align_corners=False)
        if self.precision == 'fp16':
            image = image.half()
        # RN50 主干完全冻结，仅推理。
        with torch.no_grad():
            rn50_features = self.rn50_model.encode_image(image)
        # OpenCLIP 不同版本返回值结构不一，这里统一压成 (B, dim) 张量。
        if isinstance(rn50_features, (list, tuple)):
            if len(rn50_features) == 1 and rn50_features[0].dim() == 2:
                rn50_features = rn50_features[0]
            elif rn50_features and rn50_features[0].dim() == 1:
                rn50_features = torch.stack(list(rn50_features), dim=0)
            else:
                rn50_features = rn50_features[0]
        return F.normalize(rn50_features, dim=-1)

    def encode_rn50_local(self, image):
        # 直接取 layer4 的空间特征图（不经过 attention pool），
        # reshape 成 (B, num_patch, dim) 的 token 序列，用于 local_topk / guided_vit 模式。
        if self.rn50_model is None:
            raise RuntimeError('encode_rn50_local requires --rn50-visual-fusion True')
        image = F.interpolate(image.float(), size=(224, 224), mode='bilinear', align_corners=False)
        if self.precision == 'fp16':
            image = image.half()
        visual = self.rn50_model.visual
        with torch.no_grad():
            x = visual.stem(image)
            x = visual.layer1(x)
            x = visual.layer2(x)
            x = visual.layer3(x)
            x = visual.layer4(x)
            tokens = x.flatten(2).transpose(1, 2)
        return F.normalize(tokens, dim=-1)

    def build_rn50_gallery(self, rn50_features, rn50_local_features=None):
        # 把所有正常样本的 RN50 特征拼成 memory bank。
        # 测试时计算 1 - cosine 距离的最小值作为异常分数（PatchCore 风格）。
        gallery = F.normalize(rn50_features, dim=-1)
        if self.rn50_gallery.shape[0] != gallery.shape[0] or self.rn50_gallery.shape[1] != gallery.shape[1]:
            self.rn50_gallery = gallery.new_zeros(gallery.shape)
        self.rn50_gallery.copy_(gallery.to(self.rn50_gallery.dtype))
        if rn50_local_features is not None:
            # 局部 token gallery：所有正常样本的 patch 全部拉平，便于 patch 级最近邻搜索。
            b, n, d = rn50_local_features.shape
            local_gallery = F.normalize(rn50_local_features.reshape(-1, d), dim=-1)
            if (self.rn50_local_gallery.shape[0] != local_gallery.shape[0]
                    or self.rn50_local_gallery.shape[1] != local_gallery.shape[1]):
                self.rn50_local_gallery = local_gallery.new_zeros(local_gallery.shape)
            self.rn50_local_gallery.copy_(local_gallery.to(self.rn50_local_gallery.dtype))

    def calculate_rn50_global_score(self, image):
        # 全局打分：测试图 RN50 embedding 到正常 gallery 的最近余弦距离。
        rn50_features = self.encode_rn50_image(image)
        score, _ = (1.0 - rn50_features @ self.rn50_gallery.t()).min(dim=-1)
        return score / 2.0  # 缩放到大致 [0, 1] 区间。

    def calculate_rn50_local_map(self, image):
        # 局部 patch 距离图：每个 RN50 patch 找最近的正常 patch，得到 (B, 1, side, side) 的距离图。
        local_features = self.encode_rn50_local(image)
        score, _ = (1.0 - local_features @ self.rn50_local_gallery.t()).min(dim=-1)
        score = score / 2.0
        side = int(round(score.shape[1] ** 0.5))
        return score.reshape((image.shape[0], side, side)).unsqueeze(1)

    def calculate_rn50_local_score(self, image):
        # 取 patch 距离图 top-k 均值作为图像级 RN50 分数。
        local_map = self.calculate_rn50_local_map(image)
        flat = local_map.flatten(1)
        topk = max(1, int(flat.shape[1] * float(self.rn50_local_topk_ratio)))
        return torch.topk(flat, k=topk, dim=1).values.mean(dim=1)

    def calculate_rn50_guided_vit_score(self, image, visual_anomaly_map):
        # guided_vit 模式：用 RN50 的局部异常图当 attention 权重，对 ViT 的 patch map 做加权聚合。
        # 思路：RN50 作为“注意力路标”，把 ViT 已经计算好的更细致的 patch 分数集中到高响应区域。
        rn50_map = self.calculate_rn50_local_map(image).to(
            device=visual_anomaly_map.device,
            dtype=visual_anomaly_map.dtype,
        )
        # 上采样到 ViT patch map 的网格。
        rn50_map = F.interpolate(
            rn50_map,
            size=visual_anomaly_map.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        # 减去最小值 + 归一化到 [0,1]，再走 softmax 形成 attention 权重。
        weights = rn50_map.flatten(1).float()
        weights = weights - weights.amin(dim=1, keepdim=True)
        scale = weights.amax(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / scale
        # 温度越小越集中关注高分 patch。
        temperature = max(1e-4, float(self.rn50_guidance_temperature))
        weights = torch.softmax(weights / temperature, dim=1).to(visual_anomaly_map.dtype)
        vit_scores = visual_anomaly_map.flatten(1)
        # 加权求和：RN50 高响应位置贡献的 ViT 分数权重大。
        return (weights * vit_scores).sum(dim=1)

    def calculate_rn50_visual_score(self, image):
        # 按 --rn50-score-mode 分发到 global / local_topk / both。
        if self.rn50_score_mode == 'global':
            return self.calculate_rn50_global_score(image)
        if self.rn50_score_mode == 'local_topk':
            return self.calculate_rn50_local_score(image)
        if self.rn50_score_mode == 'both':
            return 0.5 * (self.calculate_rn50_global_score(image) + self.calculate_rn50_local_score(image))
        raise ValueError(f'Unknown rn50_score_mode: {self.rn50_score_mode}')

    def encode_cnn_mamba_local(self, image):
        # ── CNN-Mamba 局部分支（对应 --cnn-vit-mamba-fusion） ──────────────────
        # 结构：可训练 CNN encoder + 多层 Mamba block。
        # 与冻结的 RN50 不同，这里参数参与训练，可被 cnn_mamba_align 损失驱动向 ViT 全局特征对齐。
        # 输出已 L2 归一化的 token 序列 (B, N, dim)。
        # ───────────────────────────────────────────────────────────────────
        if self.cnn_mamba_local_branch is None:
            raise RuntimeError('encode_cnn_mamba_local requires --cnn-vit-mamba-fusion True')
        return self.cnn_mamba_local_branch(image)

    def build_cnn_mamba_gallery(self, cnn_mamba_features):
        # 同时维护 patch 级 gallery 和 image 级 gallery，分别支持 patch-NN 和聚合后图像分数。
        b, n, d = cnn_mamba_features.shape
        # patch 级：所有正常样本的所有 patch 拉平。
        gallery = F.normalize(cnn_mamba_features.reshape(-1, d), dim=-1)
        if self.cnn_mamba_gallery.shape[0] != gallery.shape[0] or self.cnn_mamba_gallery.shape[1] != gallery.shape[1]:
            self.cnn_mamba_gallery = gallery.new_zeros(gallery.shape)
        self.cnn_mamba_gallery.copy_(gallery.to(self.cnn_mamba_gallery.dtype))

        # 全局级：每张图的 token 平均，再做 L2 归一化。
        global_gallery = F.normalize(cnn_mamba_features.mean(dim=1), dim=-1)
        if (self.cnn_mamba_global_gallery.shape[0] != global_gallery.shape[0]
                or self.cnn_mamba_global_gallery.shape[1] != global_gallery.shape[1]):
            self.cnn_mamba_global_gallery = global_gallery.new_zeros(global_gallery.shape)
        self.cnn_mamba_global_gallery.copy_(global_gallery.to(self.cnn_mamba_global_gallery.dtype))

    def aggregate_cnn_mamba_by_anomaly(self, local_features, patch_score):
        # 关注高异常 patch 的加权聚合：
        # 1. 只保留 patch_score 排名前 topk 比例的 token，其余 mask 成 -∞；
        # 2. softmax 形成权重（温度越小越集中），与 token 加权求和。
        # 这样得到的“图像特征”聚焦于异常区域，而非整张图均值，便于和 cnn_mamba_global_gallery 比较。
        b, n, _ = local_features.shape
        ratio = min(1.0, max(1.0 / float(n), float(self.cnn_mamba_pool_topk_ratio)))
        topk = max(1, int(n * ratio))
        masked = patch_score.float().clone()
        if topk < n:
            threshold = torch.topk(masked, k=topk, dim=1).values[:, -1:].detach()
            masked = masked.masked_fill(masked < threshold, -1e4)
        temperature = max(1e-4, float(self.cnn_mamba_pool_temperature))
        weights = torch.softmax(masked / temperature, dim=1).to(local_features.dtype)
        pooled = (weights.unsqueeze(-1) * local_features).sum(dim=1)
        return F.normalize(pooled, dim=-1), weights

    def calculate_cnn_mamba_local_score(self, image):
        # CNN-Mamba 分支的两个输出：
        # 1) patch_map：每个 patch 到 cnn_mamba_gallery 的最近距离图，可与 ViT patch map 融合；
        # 2) image_score：先按 patch_map 做异常加权聚合得到 pooled 特征，
        #                 再到 cnn_mamba_global_gallery 找最近距离作为图像级分数。
        local_features = self.encode_cnn_mamba_local(image)
        score, _ = (1.0 - local_features @ self.cnn_mamba_gallery.t()).min(dim=-1)
        score = score / 2.0
        patch_map = score.reshape((image.shape[0], self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        pooled_feature, _ = self.aggregate_cnn_mamba_by_anomaly(local_features, score)
        image_score, _ = (1.0 - pooled_feature @ self.cnn_mamba_global_gallery.t()).min(dim=-1)
        image_score = image_score / 2.0
        return image_score, patch_map

    def _adapt_visual_features(self, image_features):
        # 把每路视觉特征通过对应维度的 ResidualVisualAdapter。
        # CLIP encode_image 返回多路特征（cls / token / 两路 patch map），
        # 不同路维度不一定相同，因此 visual_adapters 是一个按 dim 索引的 ModuleDict。
        # 没有匹配 adapter 的路直接保持不变。
        if self.visual_adapters is None:
            return image_features
        adapted_features = []
        for feature in image_features:
            key = str(feature.shape[-1])
            if key not in self.visual_adapters:
                adapted_features.append(feature)
            else:
                adapted_features.append(self.visual_adapters[key](feature))
        return adapted_features

    def encode_image(self, image: torch.Tensor):

        # CLIP 图像编码器冻结；如果启用 visual adapter，只让 adapter 接收梯度。
        if self.precision == "fp16":
            image = image.half()

        # 项目内的 CLIP encode_image 返回多路视觉特征，
        # 通常包括全局特征、局部 token 特征以及两路 patch feature map。
        grad_enabled = self.use_visual_lora
        with torch.set_grad_enabled(grad_enabled):
            image_features = self.model.encode_image(image)
        image_features = self._adapt_visual_features(image_features)

        # 对每一路视觉特征做 L2 归一化。
        # 归一化后，点积就近似等价于余弦相似度。
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        # 直接编码 tokenized 文本。
        # 当前主流程更多使用 encode_text_embedding，因为 prompt 是 embedding 形式拼出来的。
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features

    def encode_text_embedding(self, text_embedding, original_tokens):
        # 编码已经拼接好的 prompt embedding。
        # text_embedding 是 PromptLearner.forward() 输出的 embedding 序列；
        # original_tokens 用来告诉 CLIP 每条文本的 token 结构和结束位置。
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_text_feature_gallery(self):
        # 根据当前 PromptLearner 的参数，重新生成正常/异常文本特征。
        # 训练过程中 normal_ctx / abnormal_ctx 会变化，
        # 因此每个 epoch 结束后都需要重新 build text_features。
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()

        if self.version == "V1":
            # V1：直接将全部正常 prompt 编码成文本特征；
            # 同时将全部异常 prompt 编码成文本特征。
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features_handle = self.encode_text_embedding(
                abnormal_text_embeddings_handle,
                self.tokenized_abnormal_prompts_handle,
            )
            abnormal_text_features_learned = self.encode_text_embedding(
                abnormal_text_embeddings_learned,
                self.tokenized_abnormal_prompts_learned,
            )
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
        elif self.version == "V2":
            # V2：逐条 prompt 编码并归一化，当前默认不使用。
            abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        # 多条正常 prompt 求平均，得到一个正常文本原型。
        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)

        if self.text_prototype_mode in {'grouped_max', 'grouped_mean', 'grouped_meanmax', 'grouped_softmax'} and self.prompt_learner.abnormal_handle_group_slices:
            # grouped_*：保留多个异常形态原型，而不是把 burst/chirp/dsss 平均成一个异常中心。
            # 当前分组原型只使用手工异常 prompt，以最大限度保留明确的形态语义。
            group_features = []
            for _, start, end in self.prompt_learner.abnormal_handle_group_slices:
                group_feature = abnormal_text_features_handle[start:end].mean(dim=0, keepdim=True)
                group_features.append(group_feature)
            text_features = torch.cat([avr_normal_text_features] + group_features, dim=0)
        else:
            # single：多条异常 prompt 求平均，得到一个异常文本原型。
            avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)
            text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)

        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2, global_features=None):
        # 构建正常图像 patch 特征库。
        # 输入 features1/features2 通常来自正常训练样本：
        # cls_feature, _, feature_map1, feature_map2 = model.encode_image(data)
        #
        # features shape: (batch, num_patch, dim)
        # gallery shape:  (batch * num_patch, dim)
        b1, n1, d1 = features1.shape

        # 将所有正常样本的所有 patch 拉平成一个大 gallery，并做 L2 归一化。
        gallery1 = F.normalize(features1.reshape(-1, d1), dim=-1)

        # 如果实际 gallery 大小和初始化大小不同，则重新分配。
        # 这可以兼容不同 k-shot 或不同输入网格大小。
        if self.feature_gallery1.shape[0] != gallery1.shape[0] or self.feature_gallery1.shape[1] != gallery1.shape[1]:
            self.feature_gallery1 = gallery1.new_zeros(gallery1.shape)
        self.feature_gallery1.copy_(gallery1)

        b2, n2, d2 = features2.shape
        gallery2 = F.normalize(features2.reshape(-1, d2), dim=-1)
        if self.feature_gallery2.shape[0] != gallery2.shape[0] or self.feature_gallery2.shape[1] != gallery2.shape[1]:
            self.feature_gallery2 = gallery2.new_zeros(gallery2.shape)
        self.feature_gallery2.copy_(gallery2)

        if global_features is not None:
            self.build_global_normal_distribution(global_features)

    def build_global_normal_distribution(self, global_features):
        # 把所有正常样本的全局特征拟合成一个对角高斯：
        # - normal_global_mean   : 各维度均值（未归一化）；
        # - normal_global_center : 均值再 L2 归一化，便于与单位向量做余弦距离；
        # - normal_global_var    : 各维度方差，clamp 到 ridge 之上避免除零。
        # 用于 cls_score_mode = normal_center / normal_mahalanobis 等模式。
        normal_features = F.normalize(global_features.float(), dim=-1)
        mean = normal_features.mean(dim=0)
        center = F.normalize(mean, dim=0)
        var = normal_features.var(dim=0, unbiased=False).clamp_min(self.normal_dist_ridge)
        self.normal_global_mean.copy_(mean.to(self.normal_global_mean.dtype))
        self.normal_global_center.copy_(center.to(self.normal_global_center.dtype))
        self.normal_global_var.copy_(var.to(self.normal_global_var.dtype))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # 计算“文本异常分数”。
        #
        # 核心思想：
        # 将图像特征分别和两个文本原型做相似度：
        #   text_features[0] = normal 文本原型
        #   text_features[1] = abnormal 文本原型
        # 再经过 softmax 得到 [正常概率, 异常概率]。
        # 其中异常概率就是文本异常分数。

        # CLIP 的 logit_scale 温度参数，用来放大图文相似度。
        # 温度越大，softmax 后的概率分布越尖锐。
        t = self.model.logit_scale
        use_grouped_scoring = (
            self.text_prototype_mode in {'grouped_max', 'grouped_mean', 'grouped_meanmax', 'grouped_softmax'}
            and self.text_features.shape[0] > 2
        )

        # batch size。
        N = visual_features[1].shape[0]

        if task == 'seg':
            # seg 分支使用局部 token 特征。
            # 每个 patch/token 都会和 normal/abnormal 文本原型做相似度，
            # 因此输出的是一张 patch 级文本异常图。
            token_features = visual_features[1]

            logits = t * token_features @ self.text_features.T
            if use_grouped_scoring:
                # 多异常原型模式：每个 patch 分别和 burst/chirp/dsss 原型比较。
                normal_logit = logits[:, :, 0:1]
                grouped_margin = logits[:, :, 1:] - normal_logit
                if self.text_prototype_mode == 'grouped_mean':
                    local_abnormality_score = grouped_margin.mean(dim=-1).sigmoid()
                elif self.text_prototype_mode == 'grouped_softmax':
                    grouped_weights = grouped_margin.softmax(dim=-1)
                    local_fused_margin = (grouped_weights * grouped_margin).sum(dim=-1)
                    local_abnormality_score = local_fused_margin.sigmoid()
                elif self.text_prototype_mode == 'grouped_meanmax':
                    proto_scores = grouped_margin.sigmoid()
                    local_abnormality_score = 0.5 * proto_scores.mean(dim=-1) + 0.5 * proto_scores.max(dim=-1).values
                else:
                    local_abnormality_score = grouped_margin.max(dim=-1).values.sigmoid()
            else:
                # shape 近似为：
                # token_features: (N, num_patch, dim)
                # self.text_features.T: (dim, 2)
                # 输出: (N, num_patch, 2)，最后一维是 [normal_score, abnormal_score]。
                local_normality_and_abnormality_score = logits.softmax(dim=-1)

                # 取第 1 类，也就是 abnormal 概率。
                local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            # reshape 成二维 patch 网格：
            # (N, grid_h * grid_w) -> (N, 1, grid_h, grid_w)
            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # cls 分支使用全局图像特征。
            # 它直接输出每张图的图像级 abnormal 概率。
            global_feature = visual_features[0]

            logits = t * global_feature @ self.text_features.T
            if use_grouped_scoring:
                # 多异常原型模式：
                # grouped_max: final_score = sigmoid(max_k(sim(image, abnormal_k) - sim(image, normal)))
                # grouped_mean: final_score = sigmoid(mean_k(sim(image, abnormal_k) - sim(image, normal)))
                # grouped_softmax: final_score = sigmoid(sum_k softmax(margin_k) * margin_k)
                # grouped_meanmax: final_score = 0.5 * mean(proto_scores) + 0.5 * max(proto_scores)
                normal_logit = logits[:, 0:1]
                grouped_margin = logits[:, 1:] - normal_logit
                if self.text_prototype_mode == 'grouped_mean':
                    global_abnormality_score = grouped_margin.mean(dim=-1).sigmoid()
                elif self.text_prototype_mode == 'grouped_softmax':
                    grouped_weights = grouped_margin.softmax(dim=-1)
                    global_fused_margin = (grouped_weights * grouped_margin).sum(dim=-1)
                    global_abnormality_score = global_fused_margin.sigmoid()
                elif self.text_prototype_mode == 'grouped_meanmax':
                    proto_scores = grouped_margin.sigmoid()
                    global_abnormality_score = 0.5 * proto_scores.mean(dim=-1) + 0.5 * proto_scores.max(dim=-1).values
                else:
                    global_abnormality_score = grouped_margin.max(dim=-1).values.sigmoid()
            else:
                # global_feature: (N, dim)
                # self.text_features.T: (dim, 2)
                # 输出: (N, 2)，表示每张图对应 [normal_prob, abnormal_prob]。
                global_normality_and_abnormality_score = logits.softmax(dim=-1)

                # 取 abnormal 概率作为图像级异常分数。
                global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        # 计算“视觉异常分数”。
        #
        # 核心思想：
        # 当前测试图像的每个 patch 特征，去正常训练样本的 feature_gallery 中找最近邻。
        # 如果某个 patch 和所有正常 patch 都不像，则它的异常分数高。
        #
        # 因为 encode_image 已经对特征做 L2 归一化，
        # 所以矩阵乘法 visual_feature @ gallery.T 等价于余弦相似度。
        N = visual_features[1].shape[0]

        # 沿 gallery 维度分块计算，避免一次性物化 (B, P, N·P) 的中间张量。
        # 最终结果和单次大矩阵乘相同：每个 probe patch 取与所有 gallery patch
        # 的最小余弦距离。chunk_size 是经验值，单块矩阵控制在 ~1 GiB 以内。
        score1 = self._min_cosine_dist_chunked(
            visual_features[2], self.feature_gallery1, self.visual_gallery_chunk_size
        )
        score1 = score1 / 2.0

        score2 = self._min_cosine_dist_chunked(
            visual_features[3], self.feature_gallery2, self.visual_gallery_chunk_size
        )
        score2 = score2 / 2.0

        # 两路视觉异常分数取平均。
        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        # reshape 成 patch 级异常图：
        # (N, grid_h * grid_w) -> (N, 1, grid_h, grid_w)
        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    @staticmethod
    def _min_cosine_dist_chunked(probe, gallery, chunk_size=1024):
        """对每个 probe patch 求与 gallery 中最相似 patch 的余弦距离。

        probe   shape: (B, P, D)
        gallery shape: (M, D)
        return  shape: (B, P)
        """
        chunk_size = max(1, int(chunk_size))
        if gallery.shape[0] <= chunk_size:
            sim, _ = (probe @ gallery.t()).max(dim=-1)
            return 1.0 - sim

        best_sim = None
        for start in range(0, gallery.shape[0], chunk_size):
            chunk = gallery[start:start + chunk_size]
            sim_chunk, _ = (probe @ chunk.t()).max(dim=-1)
            best_sim = sim_chunk if best_sim is None else torch.maximum(best_sim, sim_chunk)
        return 1.0 - best_sim

    def calculate_dense_mask_logits(self, visual_features):
        if self.dense_mask_head is None:
            raise RuntimeError('calculate_dense_mask_logits requires --dense-mask-branch True')
        return self.dense_mask_head(visual_features[2], visual_features[3], self.grid_size)

    def calculate_dense_mask_score(self, visual_features):
        raw_logits, post_logits = self.calculate_dense_mask_logits(visual_features)
        dense_map = 0.5 * (torch.sigmoid(raw_logits) + torch.sigmoid(post_logits))
        flat = dense_map.flatten(1)
        topk = max(1, int(flat.shape[1] * self.visual_topk_ratio))
        image_score = torch.topk(flat, k=topk, dim=1).values.mean(dim=1)
        return image_score, dense_map

    def calculate_text_aligned_dense_logits(self, visual_features):
        if self.text_aligned_dense_head is None:
            raise RuntimeError('calculate_text_aligned_dense_logits requires --text-aligned-dense True')
        return self.text_aligned_dense_head(
            visual_features[2],
            visual_features[3],
            self.text_features,
            self.grid_size,
            self.model.logit_scale,
        )

    def summarize_text_aligned_map(self, ta_map):
        flat = ta_map.flatten(1)
        topk = max(1, int(flat.shape[1] * self.visual_topk_ratio))
        topk_score = torch.topk(flat, k=topk, dim=1).values.mean(dim=1)
        max_score = flat.max(dim=1).values
        return topk_score, max_score

    def calculate_text_aligned_dense_score(self, visual_features):
        raw_logits, post_logits = self.calculate_text_aligned_dense_logits(visual_features)
        ta_map = 0.5 * (torch.sigmoid(raw_logits) + torch.sigmoid(post_logits))
        image_score, _ = self.summarize_text_aligned_map(ta_map)
        return image_score, ta_map

    def set_text_aligned_normal_calibration(self, visual_features):
        if self.text_aligned_dense_head is None:
            return
        _, ta_map = self.calculate_text_aligned_dense_score(visual_features)
        topk_score, max_score = self.summarize_text_aligned_map(ta_map)

        def _stats(score):
            score = score.detach().float().flatten()
            q = torch.quantile(score, self.ta_normal_quantile)
            median = torch.quantile(score, 0.5)
            scale = (q - median).abs().clamp_min(self.ta_normal_excess_scale_floor)
            return q.detach(), scale.detach()

        self.ta_normal_topk_threshold, self.ta_normal_topk_scale = _stats(topk_score)
        self.ta_normal_max_threshold, self.ta_normal_max_scale = _stats(max_score)

    def calculate_text_aligned_normal_excess(self, ta_map, use_max=False):
        topk_score, max_score = self.summarize_text_aligned_map(ta_map)
        if use_max:
            score = max_score
            threshold = self.ta_normal_max_threshold
            scale = self.ta_normal_max_scale
        else:
            score = topk_score
            threshold = self.ta_normal_topk_threshold
            scale = self.ta_normal_topk_scale
        if threshold is None or scale is None:
            return torch.zeros_like(score)
        threshold = threshold.to(device=score.device, dtype=score.dtype)
        scale = scale.to(device=score.device, dtype=score.dtype).clamp_min(self.ta_normal_excess_scale_floor)
        return ((score - threshold) / scale).clamp(min=0.0, max=1.0)

    @staticmethod
    def harmonic_score_tensors(*scores, eps=1e-6):
        safe = [s.float().clamp_min(eps) for s in scores]
        denom = sum(1.0 / s for s in safe)
        return float(len(safe)) / denom

    def calculate_normal_distribution_score(self, visual_features, mode):
        # 基于正常分布的图像级异常分数。
        # mode = 'center'      ：到正常类别中心的余弦距离；
        # mode = 'mahalanobis' ：标准化后的简化马氏距离（对角协方差）。
        global_feature = F.normalize(visual_features[0].float(), dim=-1)
        if mode == 'center':
            center = F.normalize(self.normal_global_center.float(), dim=0)
            score = (1.0 - global_feature @ center) / 2.0
        elif mode == 'mahalanobis':
            mean = self.normal_global_mean.float()
            var = self.normal_global_var.float().clamp_min(self.normal_dist_ridge)
            score = torch.sqrt(((global_feature - mean) ** 2 / var).mean(dim=-1))
        else:
            raise ValueError(f'Unsupported normal distribution score mode: {mode}')
        return score.detach().cpu().numpy()

    def aggregate_visual_image_score(self, visual_anomaly_map):
        # 把视觉 patch 异常图聚合成图像级 visual 分数。
        # 关键步骤：
        # 1. flatten 后取 top-k 个最高 patch 的均值（mean_topk），抑制极端单点噪声；
        # 2. 也保留全图最大值 max_score，给 visual_topk_max / visual_topk_freq 模式用。
        # 频率位置加权（visual_freq_position_weight）：
        #   把频率轴 |y - 0.5| 当成位置先验，强调远离中心频率的位置（边带异常）。
        patch_scores = visual_anomaly_map.squeeze(1)
        n, h, w = patch_scores.shape
        flat_scores = patch_scores.reshape(n, -1)

        topk = max(1, int(flat_scores.shape[1] * self.visual_topk_ratio))
        topk_scores, _ = torch.topk(flat_scores, k=topk, dim=1)
        mean_topk = topk_scores.mean(dim=1)
        max_score = flat_scores.max(dim=1).values

        if self.visual_freq_position_weight > 0:
            # 频率位置先验：离 0.5（中心频率）越远权重越大。
            freq_axis = torch.linspace(0.0, 1.0, steps=h, device=patch_scores.device, dtype=patch_scores.dtype)
            freq_weights = 1.0 + self.visual_freq_position_weight * (freq_axis - 0.5).abs() * 2.0
            weighted_map = patch_scores * freq_weights.view(1, h, 1)
            weighted_flat = weighted_map.reshape(n, -1)
            weighted_topk, _ = torch.topk(weighted_flat, k=topk, dim=1)
            mean_topk = weighted_topk.mean(dim=1)
            max_score = weighted_flat.max(dim=1).values

        # 模式选择：单纯 top-k 还是 top-k + max 加权。
        if self.cls_score_mode == 'visual_topk':
            visual_score = mean_topk
        elif self.cls_score_mode == 'visual_topk_max':
            visual_score = self.visual_score_beta * mean_topk + self.visual_score_gamma * max_score
        else:
            visual_score = mean_topk

        return visual_score.detach().cpu().numpy()

    def fuse_cls_scores(self, textual_anomaly, visual_anomaly_map, visual_features=None):
        # 图像级最终分数融合的总入口（不包括 RN50 / CNN-Mamba / 可学习融合头三种特殊路径，
        # 那三种在 score_cached 中独立处理）。
        # 各 cls_score_mode 行为：
        #   text_only                : 直接返回文本异常分数（主方案）；
        #   normal_center / mahala.. : 仅用正常分布距离作为分数；
        #   text_normal_*            : α·文本 + β·正常分布距离；
        #   visual_topk              : α·文本 + β·visual top-k 均值；
        #   visual_topk_max / freq   : α·文本 + β·top-k + γ·max。
        if self.cls_score_mode == 'text_only':
            return textual_anomaly

        if self.cls_score_mode in {'normal_center', 'normal_mahalanobis', 'text_normal_center', 'text_normal_mahalanobis'}:
            if visual_features is None:
                # 没有视觉特征（缓存路径未传）时回退到文本分数。
                return textual_anomaly
            dist_mode = 'center' if 'center' in self.cls_score_mode else 'mahalanobis'
            normal_score = self.calculate_normal_distribution_score(visual_features, dist_mode)
            if self.cls_score_mode.startswith('text_'):
                return self.visual_score_alpha * textual_anomaly + self.visual_score_beta * normal_score
            return normal_score

        # 其余模式都基于 visual top-k 聚合。
        visual_score = self.aggregate_visual_image_score(visual_anomaly_map)

        if self.cls_score_mode == 'visual_topk':
            return self.visual_score_alpha * textual_anomaly + self.visual_score_beta * visual_score

        if self.cls_score_mode in {'visual_topk_max', 'visual_topk_freq'}:
            # 这里直接重新算一遍 max（aggregate_visual_image_score 内部已用过，但只返回 top-k 聚合）。
            max_score = visual_anomaly_map.squeeze(1).reshape(visual_anomaly_map.shape[0], -1).max(dim=1).values
            max_score = max_score.detach().cpu().numpy()
            return (
                self.visual_score_alpha * textual_anomaly
                + self.visual_score_beta * visual_score
                + self.visual_score_gamma * max_score
            )

        return textual_anomaly

    def score_cached(self, visual_features, task='cls', return_raw_map=False, image=None):
        # 使用已经缓存好的视觉特征计算异常分数。
        #
        # 当前旧分支中，CLIP 图像编码器冻结，训练过程中图像特征不变。
        # 因此 train_cls.py 会提前缓存测试集 visual_features，
        # 每个 epoch 评估时只重新计算文本特征和异常分数，避免重复跑 CLIP 图像编码器。
        if task != 'cls':
            raise ValueError(f"score_cached only supports 'cls', got {task!r}")

        # cls 的图像级分数先按原 PromptAD 路径计算。
        textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

        # 原视觉分支生成 patch 级异常图，用于 score map / 可视化 / 原有 score fusion。
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        image_scores = np.asarray(
            self.fuse_cls_scores(textual_anomaly, visual_anomaly_map, visual_features),
            dtype=np.float32,
        )
        if self.dense_mask_head is not None:
            dense_image_score, dense_map = self.calculate_dense_mask_score(visual_features)
            dense_image_score_np = dense_image_score.detach().cpu().numpy()
            if self.cls_score_mode == 'dense_only':
                image_scores = dense_image_score_np
                visual_anomaly_map = dense_map.to(visual_anomaly_map.device, dtype=visual_anomaly_map.dtype)
            elif self.cls_score_mode == 'dense_fusion':
                image_scores = image_scores + float(self.dense_mask_score_alpha) * dense_image_score_np
                visual_anomaly_map = dense_map.to(visual_anomaly_map.device, dtype=visual_anomaly_map.dtype)
            elif self.cls_score_mode == 'dense_map_only':
                # Keep the original PromptAD image score, but use the VCP/dense map for pixel metrics.
                visual_anomaly_map = dense_map.to(visual_anomaly_map.device, dtype=visual_anomaly_map.dtype)
        if self.text_aligned_dense_head is not None:
            ta_image_score, ta_map = self.calculate_text_aligned_dense_score(visual_features)
            ta_image_score_np = ta_image_score.detach().cpu().numpy()
            ta_map_for_eval = ta_map.to(visual_anomaly_map.device, dtype=visual_anomaly_map.dtype)
            if self.cls_score_mode == 'ta_map_only':
                visual_anomaly_map = ta_map_for_eval
            elif self.cls_score_mode == 'ta_image_only':
                image_scores = ta_image_score_np
                visual_anomaly_map = ta_map_for_eval
            elif self.cls_score_mode == 'ta_topk':
                image_scores = image_scores + float(self.text_aligned_dense_score_beta) * ta_image_score_np
                visual_anomaly_map = ta_map_for_eval
            elif self.cls_score_mode in {'ta_max', 'ta_max_only'}:
                ta_max_score_np = ta_map.flatten(1).max(dim=1).values.detach().cpu().numpy()
                if self.cls_score_mode == 'ta_max_only':
                    image_scores = ta_max_score_np
                else:
                    image_scores = image_scores + float(self.text_aligned_dense_score_beta) * ta_max_score_np
                visual_anomaly_map = ta_map_for_eval
            elif self.cls_score_mode in {'ta_norm_excess', 'ta_norm_max_excess'}:
                excess = self.calculate_text_aligned_normal_excess(
                    ta_map, use_max=self.cls_score_mode == 'ta_norm_max_excess'
                )
                image_scores = image_scores + float(self.text_aligned_dense_score_beta) * excess.detach().cpu().numpy()
                visual_anomaly_map = ta_map_for_eval
            elif self.cls_score_mode == 'ta_harmonic':
                textual_t = torch.as_tensor(image_scores, device=ta_image_score.device, dtype=ta_image_score.dtype)
                max_map = ta_map.flatten(1).max(dim=1).values
                image_scores = self.harmonic_score_tensors(textual_t, ta_image_score, max_map).detach().cpu().numpy()
                visual_anomaly_map = ta_map_for_eval
        if self.rn50_model is not None:
            if image is None:
                raise ValueError('score_cached requires image tensors when --rn50-visual-fusion is enabled')
            if self.rn50_fusion_mode == 'guided_vit':
                rn50_score = self.calculate_rn50_guided_vit_score(image, visual_anomaly_map)
            else:
                rn50_score = self.calculate_rn50_visual_score(image)
            image_scores = image_scores + float(self.rn50_visual_beta) * rn50_score.detach().cpu().numpy()
        if self.cnn_mamba_local_branch is not None:
            if image is None:
                raise ValueError('score_cached requires image tensors when --cnn-vit-mamba-fusion is enabled')
            local_image_score, local_patch_map = self.calculate_cnn_mamba_local_score(image)
            image_scores = image_scores + float(self.cnn_mamba_beta) * local_image_score.detach().cpu().numpy()
            if float(self.cnn_mamba_map_beta) > 0:
                visual_anomaly_map = visual_anomaly_map + float(self.cnn_mamba_map_beta) * local_patch_map.to(visual_anomaly_map.device)
        elif self.score_fusion_head is not None and self.rn50_model is None:
            textual_anomaly_t = torch.as_tensor(textual_anomaly, device=visual_anomaly_map.device, dtype=visual_anomaly_map.dtype)
            image_scores = torch.sigmoid(self.calculate_learnable_fusion_logit(textual_anomaly_t, visual_anomaly_map))
            image_scores = image_scores.detach().cpu().numpy()
        anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w),
                                    mode='bilinear', align_corners=False)

        # 转成 numpy list，供 metric_cal_img 和可视化函数使用。
        am_pix = anomaly_map.detach().squeeze(1).cpu().numpy()
        am_pix_list = [am_pix[i] for i in range(am_pix.shape[0])]
        am_img_list = [image_scores[i] for i in range(len(image_scores))]

        if return_raw_map:
            raw_map_np = visual_anomaly_map.detach().squeeze(1).cpu().numpy()  # [N, grid_h, grid_w]
            raw_map_list = [raw_map_np[i] for i in range(raw_map_np.shape[0])]
            return am_img_list, am_pix_list, raw_map_list, textual_anomaly

        return am_img_list, am_pix_list

    def forward(self, images, task):

        # 首先通过 CLIP 图像编码器得到多路视觉特征。
        # visual_features[0]：全局图像特征，主要用于 cls 文本异常分数；
        # visual_features[1]：局部 token 特征，主要用于 seg 文本异常图；
        # visual_features[2]/[3]：patch feature map，主要用于视觉 memory bank 异常图。
        visual_features = self.encode_image(images)

        if task == 'seg':
            # seg 任务需要输出像素级异常图。
            # 文本分支：每个 patch 和 normal/abnormal 文本原型比较，得到文本异常图。
            textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')

            # 视觉分支：每个 patch 和正常 gallery 比较，得到视觉异常图。
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

            # 融合文本异常图和视觉异常图。
            # 当前使用的是调和式融合，不是普通加权求和。
            # 直观理解：两条分支都认为异常的位置，融合后更容易得到高响应。
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)
            # anomaly_map = 0.5 * (textual_anomaly_map + visual_anomaly_map)
            # anomaly_map = visual_anomaly_map
            # anomaly_map = textual_anomaly_map

            # 上采样到目标输出尺寸。
            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='nearest')

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                # 使用高斯滤波平滑异常图，减少 patch 网格带来的块状感。
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            # cls 任务需要输出图像级异常分数。
            # 当前实现中，图像级分数直接来自文本异常分数。
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

            # 同时计算视觉异常图。
            # 注意：这里的视觉异常图没有被聚合成图像级分数参与 Image-AUROC，
            # 主要用于 score map 可视化和辅助分析。
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

            # 将 patch 网格异常图插值到输出分辨率。
            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])

            am_img_list = []
            for i in range(textual_anomaly.shape[0]):
                # am_img_list 是每张图的异常分数。
                # 该分数会用于 Image-AUROC 计算。
                am_img_list.append(textual_anomaly[i])

            return am_img_list, am_pix_list
        else:
            assert 'task error'

    def train_mode(self):
        # 训练模式。
        # 旧分支中训练脚本主要优化 PromptLearner；
        # 这里调用 self.model.train() 会影响 CLIP 内部 BN/Dropout 等状态。
        self.model.train()

    def eval_mode(self):
        # 评估模式。
        # 训练开始构建 gallery、缓存测试特征以及测试阶段都会调用该模式。
        self.model.eval()
