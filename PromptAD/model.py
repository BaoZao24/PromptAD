import torch
import random
import torch.nn as nn
import cv2
import numpy as np
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image
from scipy.ndimage import gaussian_filter

from .CLIPAD import SimpleTokenizer as _Tokenizer

# 本地 tokenizer：这里仅创建对象备用，不会自动补 padding / SOS / EOS。
_tokenizer = _Tokenizer()

valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms


# CLIP 预训练时使用的图像归一化参数。
# 当前项目把频谱图转成 RGB 后送入 CLIP，因此沿用 CLIP 的 mean/std。
mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]
spectrogram_mean_train = [0.5, 0.5, 0.5]
spectrogram_std_train = [0.5, 0.5, 0.5]


def _convert_to_rgb(image):
    # CLIP 的视觉编码器需要 3 通道 RGB 输入。
    return image.convert('RGB')


class SpectrogramGradientChannels:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _safe_normalize(channel):
        scale = channel.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return channel / scale

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))

        time_grad = torch.zeros_like(gray)
        time_grad[:, :, 1:] = (gray[:, :, 1:] - gray[:, :, :-1]).abs()

        freq_grad = torch.zeros_like(gray)
        freq_grad[:, 1:, :] = (gray[:, 1:, :] - gray[:, :-1, :]).abs()

        return torch.cat([
            gray,
            self._safe_normalize(time_grad),
            self._safe_normalize(freq_grad),
        ], dim=0)


class LogPowerChannels:
    def __init__(self, gain=9.0):
        self.gain = gain
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        log_power = torch.log1p(self.gain * gray) / torch.log1p(torch.tensor(self.gain, dtype=gray.dtype))
        return log_power.repeat(3, 1, 1)


class DSSSStatisticalChannels:
    def __init__(self, kernel_size=9):
        self.kernel_size = kernel_size
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

        pad = self.kernel_size // 2
        local_mean = F.avg_pool2d(gray.unsqueeze(0), self.kernel_size, stride=1, padding=pad)
        local_sq_mean = F.avg_pool2d((gray * gray).unsqueeze(0), self.kernel_size, stride=1, padding=pad)
        local_var = (local_sq_mean - local_mean * local_mean).clamp_min(0).sqrt().squeeze(0)

        return torch.cat([
            gray,
            self._minmax(residual),
            self._minmax(local_var),
        ], dim=0)


class DSSSEnergySmoothChannels:
    def __init__(self, gain=9.0, kernel_size=17):
        self.gain = gain
        self.kernel_size = kernel_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        log_power = torch.log1p(self.gain * gray) / torch.log1p(torch.tensor(self.gain, dtype=gray.dtype))

        pad = self.kernel_size // 2
        smoothed = F.avg_pool2d(gray.unsqueeze(0), self.kernel_size, stride=1, padding=pad).squeeze(0)

        return torch.cat([gray, log_power, smoothed], dim=0)


class DSSSLowFreqBandChannels:
    def __init__(self, kernel_size=21):
        self.kernel_size = kernel_size
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))

        pad = self.kernel_size // 2
        low_freq_smooth = F.avg_pool2d(gray.unsqueeze(0), self.kernel_size, stride=1, padding=pad).squeeze(0)
        band_energy = gray.mean(dim=-1, keepdim=True).expand_as(gray)

        return torch.cat([
            gray,
            self._minmax(low_freq_smooth),
            self._minmax(band_energy),
        ], dim=0)


class DSSSEnergyProfileChannels:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        time_profile = gray.mean(dim=-2, keepdim=True).expand_as(gray)
        freq_profile = gray.mean(dim=-1, keepdim=True).expand_as(gray)

        return torch.cat([
            gray,
            self._minmax(time_profile),
            self._minmax(freq_profile),
        ], dim=0)


class DSSSRGBResidualChannels:
    def __init__(self):
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
        return torch.cat([gray, gray, self._minmax(residual)], dim=0)


class DSSSWeakResidualChannels:
    def __init__(self, alpha=0.2):
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
        return torch.cat([gray, gray, weak_residual], dim=0)


class DSSSCLAHEChannels:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        gray_pil = image.convert('L')
        gray = self.to_tensor(gray_pil)
        gray_np = np.array(gray_pil)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray_np)
        enhanced = self.to_tensor(Image.fromarray(enhanced))
        return torch.cat([gray, gray, enhanced], dim=0)


class ResidualVisualAdapter(nn.Module):
    def __init__(self, dim, bottleneck_ratio=0.25, alpha=0.2):
        super().__init__()
        hidden_dim = max(1, int(dim * bottleneck_ratio))
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature):
        original_dtype = feature.dtype
        adapted = feature.float() + self.alpha * self.net(feature.float())
        return adapted.to(original_dtype)


class PromptLearner(nn.Module):
    # PromptLearner 负责生成三类文本 prompt：
    # 1. normal_prompts：正常样本 prompt，包含可学习 normal_ctx；
    # 2. abnormal_prompts_handle：手工异常模板 prompt，来自 ad_prompts.py；
    # 3. abnormal_prompts_learned：可学习异常 prompt，包含可学习 abnormal_ctx。
    #
    # 这里的“可学习 prompt”不是普通英文单词，而是 CLIP 文本 embedding 空间中的可训练向量。
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre,
                 dataset_name=None, prompt_mode="rf", train_site=None):
        super().__init__()

        # 根据 CLIP 精度选择 prompt 向量的数据类型。
        # fp16 可以节省显存，float32 更稳定但更耗显存。
        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        # rf 模式会把 RF 任务中的信号类型解析成更适合 CLIP 的自然语言。
        # legacy 模式保留旧逻辑，便于做 ablation。
        state_anomaly1 = get_abnormal_prompt_states(
            classname, dataset_name, prompt_mode, train_site=train_site
        )
        classname = get_prompt_classname(
            classname, dataset_name, prompt_mode, train_site=train_site
        )

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
        self.normal_ctx = nn.Parameter(normal_ctx_vectors)
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)

        # 正常 prompt 模板，例如：
        # "N N N N radio frequency spectrum."
        # 其中 N 的位置后续会被 normal_ctx 替换成可学习向量。
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # 手工异常 prompt 数量。
        # 每个手工模板会和 n_pro 组 normal_ctx 组合。
        self.n_ab_handle = len(state_anomaly1)

        # 手工异常 prompt，例如：
        # "N N N N abnormal radio frequency spectrum."
        # "N N N N radio frequency spectrum with defect."
        # 注意：它仍然使用 normal_ctx 作为前缀上下文，但异常语义来自手工模板。
        abnormal_prompts_handle = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]

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

    def forward(self):

        # 生成正常 prompt 的 embedding 序列。
        # 结构为：[SOS] + normal_ctx + [classname + "." + EOS]
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

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
        abnormal_suffix_handle = self.abnormal_token_suffix_handle

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
        abnormal_suffix_learned = self.abnormal_token_suffix_learned
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
        # 对 rf_open 来说，k_shot=1 表示用 1 条 MeasRes 记录作为正常训练样本来源。
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
        self.dataset_name = kwargs.get('dataset', None)
        self.train_site = kwargs.get('train_site', None)
        self.input_mode = self._resolve_input_mode(kwargs.get('input_mode', 'auto'), self.dataset_name)
        self.cls_score_mode = kwargs.get('cls_score_mode', 'text_only')
        self.visual_topk_ratio = kwargs.get('visual_topk_ratio', 0.05)
        self.visual_score_alpha = kwargs.get('visual_score_alpha', 1.0)
        self.visual_score_beta = kwargs.get('visual_score_beta', 1.0)
        self.visual_score_gamma = kwargs.get('visual_score_gamma', 0.0)
        self.visual_freq_position_weight = kwargs.get('visual_freq_position_weight', 0.0)
        self.use_visual_adapter = kwargs.get('visual_adapter', False)
        self.adapter_bottleneck_ratio = kwargs.get('adapter_bottleneck_ratio', 0.25)
        self.adapter_alpha = kwargs.get('adapter_alpha', 0.2)
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
        if self.input_mode == 'spectral_gradient':
            self.transform = transforms.Compose(pre_resize_crop + [
                SpectrogramGradientChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'log_power':
            self.transform = transforms.Compose(pre_resize_crop + [
                LogPowerChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_statistical':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSStatisticalChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_energy_smooth':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSEnergySmoothChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_lowfreq_band':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSLowFreqBandChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_energy_profile':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSEnergyProfileChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_rgb_residual':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSRGBResidualChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_weak_residual':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSWeakResidualChannels(alpha=0.2),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_clahe':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSCLAHEChannels(),
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
            if input_mode == 'signal_adaptive':
                if dataset_name in {'burst_signal', 'chirp_signal', 'wideband_pulse'}:
                    return 'spectral_gradient'
                if dataset_name == 'dsss_signal':
                    return 'rgb'
                if dataset_name in {'spectrum', 'sample', 'deceptive_signal', 'rf_open'}:
                    return 'spectral_gradient'
                return 'rgb'
            return input_mode
        if dataset_name in {'spectrum', 'sample', 'deceptive_signal', 'burst_signal', 'dsss_signal', 'chirp_signal', 'wideband_pulse', 'rf_open'}:
            return 'spectral_gradient'
        return 'rgb'

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
            train_site=self.train_site,
        )
        self.model = model.to(self.device)
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

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None

        # CLIP 视觉分支的 patch 网格大小，例如 15×15。
        # 后续 patch 级异常图会 reshape 成这个网格。
        self.grid_size = model.visual.grid_size
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

        # text_features 保存最终用于推理的两个文本原型：
        # text_features[0] = 正常文本特征；
        # text_features[1] = 异常文本特征。
        text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)

        # 如果使用 fp16，把缓存特征也转成 half，减少显存占用。
        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()

        # 保存 tokenized prompt。
        # build_text_feature_gallery 时需要这些 token id 来调用 CLIP text encoder。
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)

    def trainable_parameters(self):
        params = list(self.prompt_learner.parameters())
        if self.visual_adapters is not None:
            params.extend(self.visual_adapters.parameters())
        if self.use_visual_lora:
            params.extend(
                p for name, p in self.model.visual.named_parameters()
                if 'lora_' in name and p.requires_grad
            )
        return params

    def _adapt_visual_features(self, image_features):
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

        # 异常文本包含两部分：
        # 1. 手工异常 prompt；
        # 2. 可学习异常 prompt。
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            # V1：直接将全部正常 prompt 编码成文本特征；
            # 同时将全部异常 prompt 编码成文本特征。
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.tokenized_abnormal_prompts)
        elif self.version == "V2":
            # V2：逐条 prompt 编码并归一化，当前默认不使用。
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
        # 多条异常 prompt 求平均，得到一个异常文本原型。
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        # 这里对所有单条文本特征做归一化，但后续真正写入的是平均后的两个原型。
        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features

        # 最终推理只保留两个文本原型：
        # 第 0 个表示 normal，第 1 个表示 abnormal。
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2):
        # 构建正常图像 patch 特征库。
        # 输入 features1/features2 通常来自正常训练样本：
        # _, _, feature_map1, feature_map2 = model.encode_image(data)
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

        # batch size。
        N = visual_features[1].shape[0]

        if task == 'seg':
            # seg 分支使用局部 token 特征。
            # 每个 patch/token 都会和 normal/abnormal 文本原型做相似度，
            # 因此输出的是一张 patch 级文本异常图。
            token_features = visual_features[1]

            # shape 近似为：
            # token_features: (N, num_patch, dim)
            # self.text_features.T: (dim, 2)
            # 输出: (N, num_patch, 2)，最后一维是 [normal_score, abnormal_score]。
            local_normality_and_abnormality_score = (t * token_features @ self.text_features.T).softmax(dim=-1)

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

            # global_feature: (N, dim)
            # self.text_features.T: (dim, 2)
            # 输出: (N, 2)，表示每张图对应 [normal_prob, abnormal_prob]。
            global_normality_and_abnormality_score = (t * global_feature @ self.text_features.T).softmax(dim=-1)

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

        # 第一路 patch 特征和 feature_gallery1 做余弦相似度。
        # 1.0 - cosine_similarity 将“相似度”转成“距离”。
        # min(dim=-1) 表示对每个测试 patch 找一个最相似的正常 patch。
        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)

        # 除以 2 是为了把余弦距离大致缩放到 [0, 1] 范围。
        score1 /= 2.0

        # 第二路 patch 特征同理。
        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        # 两路视觉异常分数取平均。
        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        # reshape 成 patch 级异常图：
        # (N, grid_h * grid_w) -> (N, 1, grid_h, grid_w)
        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def aggregate_visual_image_score(self, visual_anomaly_map):
        patch_scores = visual_anomaly_map.squeeze(1)
        n, h, w = patch_scores.shape
        flat_scores = patch_scores.reshape(n, -1)

        topk = max(1, int(flat_scores.shape[1] * self.visual_topk_ratio))
        topk_scores, _ = torch.topk(flat_scores, k=topk, dim=1)
        mean_topk = topk_scores.mean(dim=1)
        max_score = flat_scores.max(dim=1).values

        if self.visual_freq_position_weight > 0:
            freq_axis = torch.linspace(0.0, 1.0, steps=h, device=patch_scores.device, dtype=patch_scores.dtype)
            freq_weights = 1.0 + self.visual_freq_position_weight * (freq_axis - 0.5).abs() * 2.0
            weighted_map = patch_scores * freq_weights.view(1, h, 1)
            weighted_flat = weighted_map.reshape(n, -1)
            weighted_topk, _ = torch.topk(weighted_flat, k=topk, dim=1)
            mean_topk = weighted_topk.mean(dim=1)
            max_score = weighted_flat.max(dim=1).values

        if self.cls_score_mode == 'visual_topk':
            visual_score = mean_topk
        elif self.cls_score_mode == 'visual_topk_max':
            visual_score = self.visual_score_beta * mean_topk + self.visual_score_gamma * max_score
        else:
            visual_score = mean_topk

        return visual_score.detach().cpu().numpy()

    def fuse_cls_scores(self, textual_anomaly, visual_anomaly_map):
        if self.cls_score_mode == 'text_only':
            return textual_anomaly

        visual_score = self.aggregate_visual_image_score(visual_anomaly_map)

        if self.cls_score_mode == 'visual_topk':
            return self.visual_score_alpha * textual_anomaly + self.visual_score_beta * visual_score

        if self.cls_score_mode in {'visual_topk_max', 'visual_topk_freq'}:
            max_score = visual_anomaly_map.squeeze(1).reshape(visual_anomaly_map.shape[0], -1).max(dim=1).values
            max_score = max_score.detach().cpu().numpy()
            return (
                self.visual_score_alpha * textual_anomaly
                + self.visual_score_beta * visual_score
                + self.visual_score_gamma * max_score
            )

        return textual_anomaly

    def score_cached(self, visual_features, task='cls'):
        # 使用已经缓存好的视觉特征计算异常分数。
        #
        # 当前旧分支中，CLIP 图像编码器冻结，训练过程中图像特征不变。
        # 因此 train_cls.py 会提前缓存测试集 visual_features，
        # 每个 epoch 评估时只重新计算文本特征和异常分数，避免重复跑 CLIP 图像编码器。
        if task != 'cls':
            raise ValueError(f"score_cached only supports 'cls', got {task!r}")

        # cls 的图像级分数来自文本异常分数。
        textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

        # 视觉分支仍然会生成 patch 级异常图，用于 score map / 可视化。
        # 注意：当前 cls 的 Image-AUROC 主要使用 textual_anomaly，
        # 没有把 visual_anomaly_map 聚合后与 textual_anomaly 加权求和。
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        image_scores = self.fuse_cls_scores(textual_anomaly, visual_anomaly_map)
        anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w),
                                    mode='bilinear', align_corners=False)

        # 转成 numpy list，供 metric_cal_img 和可视化函数使用。
        am_pix = anomaly_map.squeeze(1).numpy()
        am_pix_list = [am_pix[i] for i in range(am_pix.shape[0])]
        am_img_list = [image_scores[i] for i in range(len(image_scores))]

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
