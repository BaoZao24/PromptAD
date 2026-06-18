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


class SpectrogramGradientChannelsV2:
    def __init__(self, sigma=1.0, percentile=0.995):
        self.sigma = sigma
        self.percentile = percentile
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        time_grad = torch.from_numpy(np.abs(time_grad)).unsqueeze(0)
        freq_grad = torch.from_numpy(np.abs(freq_grad)).unsqueeze(0)

        return torch.cat([
            gray,
            self._robust_normalize(time_grad),
            self._robust_normalize(freq_grad),
        ], dim=0)



class GrayContrastOnlyChannels:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        gray_pil = image.convert('L')
        gray_np = np.array(gray_pil)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray_np)
        enhanced = self.to_tensor(Image.fromarray(enhanced))
        return torch.cat([enhanced, enhanced, enhanced], dim=0)


class WeakResidualOnlyChannels:
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
        return torch.cat([weak_residual, weak_residual, weak_residual], dim=0)


class GradMagnitudeOnlyChannels:
    def __init__(self, sigma=1.0, percentile=0.995):
        self.sigma = sigma
        self.percentile = percentile
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(time_grad * time_grad + freq_grad * freq_grad)
        grad_mag = self._robust_normalize(torch.from_numpy(np.abs(grad_mag)).unsqueeze(0))
        return torch.cat([grad_mag, grad_mag, grad_mag], dim=0)


class MorphFusionChannels:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(time_grad * time_grad + freq_grad * freq_grad)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        grad_mag = torch.from_numpy(grad_mag).unsqueeze(0)
        return torch.cat([
            gray,
            self._robust_normalize(grad_mag),
            weak_residual,
        ], dim=0)


class MorphFusionPlusChannels:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2, grad_weight=0.35):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.grad_weight = grad_weight
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(time_grad * time_grad + freq_grad * freq_grad)
        grad_mag = self._robust_normalize(torch.from_numpy(np.abs(grad_mag)).unsqueeze(0))

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return torch.cat([
            gray,
            self._minmax(gray + self.grad_weight * grad_mag),
            weak_residual,
        ], dim=0)


class MorphFusionDualGradChannels:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return torch.cat([
            weak_residual,
            self._robust_normalize(torch.from_numpy(np.abs(time_grad)).unsqueeze(0)),
            self._robust_normalize(torch.from_numpy(np.abs(freq_grad)).unsqueeze(0)),
        ], dim=0)



class MorphFusionBalancedChannels:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2, grad_weight=0.2):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.grad_weight = grad_weight
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(time_grad * time_grad + freq_grad * freq_grad)
        grad_mag = self._robust_normalize(torch.from_numpy(np.abs(grad_mag)).unsqueeze(0))

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)
        structure_blend = self._minmax(gray + self.grad_weight * grad_mag)

        return torch.cat([
            gray,
            weak_residual,
            structure_blend,
        ], dim=0)


class MorphFusionGrayResGradChannels:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(time_grad * time_grad + freq_grad * freq_grad)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return torch.cat([
            gray,
            weak_residual,
            self._robust_normalize(torch.from_numpy(np.abs(grad_mag)).unsqueeze(0)),
        ], dim=0)




class MorphFusionGrayContrastResGradChannels:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size),
        )

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

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

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(time_grad * time_grad + freq_grad * freq_grad)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return torch.cat([
            gray_contrast,
            weak_residual,
            self._robust_normalize(torch.from_numpy(np.abs(grad_mag)).unsqueeze(0)),
        ], dim=0)


class MorphFusionGaborResidualChannels:
    def __init__(self, alpha=0.2, clahe_clip_limit=2.0, clahe_tile_grid_size=8,
                 gabor_kernel_size=31, gabor_sigma=6.0, gabor_lambda=18.0, gabor_gamma=0.5,
                 percentile=0.995):
        self.alpha = alpha
        self.percentile = percentile
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        self.gabor_kernels = [
            cv2.getGaborKernel(
                (gabor_kernel_size, gabor_kernel_size),
                gabor_sigma,
                theta,
                gabor_lambda,
                gabor_gamma,
                0,
                ktype=cv2.CV_32F,
            )
            for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        ]

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

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

        gabor_responses = [
            np.abs(cv2.filter2D(gray_np, cv2.CV_32F, kernel))
            for kernel in self.gabor_kernels
        ]
        gabor_broad = np.max(np.stack(gabor_responses, axis=0), axis=0)
        gabor_broad = self._robust_normalize(torch.from_numpy(gabor_broad).unsqueeze(0))

        return torch.cat([
            gray_contrast,
            weak_residual,
            gabor_broad,
        ], dim=0)


class MorphFusionGaborDirectionalResidualChannels:
    def __init__(self, row_weight=0.25, col_weight=0.45, clahe_clip_limit=2.0, clahe_tile_grid_size=8,
                 gabor_kernel_size=31, gabor_sigma=6.0, gabor_lambda=18.0, gabor_gamma=0.5,
                 percentile=0.995, residual_percentile=0.99):
        self.row_weight = row_weight
        self.col_weight = col_weight
        self.percentile = percentile
        self.residual_percentile = residual_percentile
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        self.gabor_kernels = [
            cv2.getGaborKernel(
                (gabor_kernel_size, gabor_kernel_size),
                gabor_sigma,
                theta,
                gabor_lambda,
                gabor_gamma,
                0,
                ktype=cv2.CV_32F,
            )
            for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        ]

    def _robust_normalize(self, channel, percentile=None):
        percentile = self.percentile if percentile is None else percentile
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

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

        row_background = np.median(gray_np, axis=1, keepdims=True)
        row_residual = np.abs(gray_np - row_background)

        col_background = np.median(gray_np, axis=0, keepdims=True)
        col_background = cv2.GaussianBlur(col_background.astype(np.float32), (31, 1), 0)
        col_residual = np.maximum(gray_np - col_background, 0.0)

        row_residual = self._robust_normalize(
            torch.from_numpy(row_residual).unsqueeze(0),
            self.residual_percentile,
        )
        col_residual = self._robust_normalize(
            torch.from_numpy(col_residual).unsqueeze(0),
            self.residual_percentile,
        )
        directional_residual = self._minmax(gray + self.row_weight * row_residual + self.col_weight * col_residual)

        gabor_responses = [
            np.abs(cv2.filter2D(gray_np, cv2.CV_32F, kernel))
            for kernel in self.gabor_kernels
        ]
        gabor_broad = np.max(np.stack(gabor_responses, axis=0), axis=0)
        gabor_broad = self._robust_normalize(torch.from_numpy(gabor_broad).unsqueeze(0))

        return torch.cat([
            gray_contrast,
            directional_residual,
            gabor_broad,
        ], dim=0)


class MorphFusionGrayResidualChannels:
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


class NormalBgDeviationChannels:
    """Compute pixel-wise deviation from training normal background distribution.

    Uses robust statistics (median + MAD) estimated from training normal samples only.
    Can be used standalone or as a channel in morph_fusion variants.

    Formula:
      median_normal[f,t] = median over N training normal images at pixel (f,t)
      MAD_normal[f,t]    = median(|I_i[f,t] - median[f,t]|) over training images
      deviation[f,t]     = |I[f,t] - median[f,t]| / (1.4826 * MAD[f,t] + eps)
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.median = None
        self.mad = None
        self.eps = 1e-6

    def set_stats(self, median_np, mad_np):
        self.median = torch.from_numpy(median_np.astype(np.float32))
        self.mad = torch.from_numpy(mad_np.astype(np.float32))

    @staticmethod
    def compute_normal_bg_stats_from_dataloader(train_dataloader, img_resize=240, img_cropsize=240):
        """Compute pixel-wise median and MAD from training normal images.

        Only uses training normal samples (label==0), consistent with normal-only training.
        Images are preprocessed identically to the model transform pipeline:
        BGR numpy -> RGB -> PIL -> Resize -> CenterCrop -> convert('L') -> grayscale.

        Returns:
            median: (H, W) numpy float32, pixel-wise median over training normals
            mad:    (H, W) numpy float32, pixel-wise median absolute deviation
        """
        all_gray = []
        print(f'Computing normal background stats from training data '
              f'(resize={img_resize}, crop={img_cropsize})...')
        for (data, mask, label, name, img_type) in tqdm(
                train_dataloader, desc='Normal bg stats', leave=False):
            for i in range(len(data)):
                if int(label[i]) != 0:
                    continue
                img_bgr = data[i].numpy()
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img = TF.resize(pil_img, (img_resize, img_resize), Image.BICUBIC)
                pil_img = TF.center_crop(pil_img, img_cropsize)
                gray_pil = pil_img.convert('L')
                gray = np.array(gray_pil, dtype=np.float32)
                if gray.max() > 1.5:
                    gray /= 255.0
                all_gray.append(gray)

        if not all_gray:
            raise RuntimeError(
                'No training normal images found for normal_bg_deviation stats computation. '
                'Check dataset and split_mode.'
            )

        stack = np.stack(all_gray, axis=0)
        median = np.median(stack, axis=0).astype(np.float32)
        mad = np.median(np.abs(stack - median), axis=0).astype(np.float32)

        print(f'  Normal bg stats computed: {len(all_gray)} images, '
              f'shape=({median.shape[0]}, {median.shape[1]}), '
              f'median(MAD)={mad.mean():.4f}')
        return median, mad

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        if self.median is None or self.mad is None:
            raise RuntimeError(
                'NormalBgDeviationChannels: stats not set. '
                'Call set_stats() before using the transform.'
            )
        median = self.median.to(gray.device)
        mad = self.mad.to(gray.device)

        abs_dev = (gray - median).abs()
        deviation = abs_dev / (1.4826 * mad + self.eps)

        deviation_np = deviation.squeeze(0).cpu().numpy()
        p2 = np.percentile(deviation_np, 2)
        p98 = np.percentile(deviation_np, 98)
        deviation = torch.clamp(deviation, float(p2), float(p98))
        deviation = (deviation - float(p2)) / max(float(p98 - p2), self.eps)

        return torch.cat([deviation, deviation, deviation], dim=0)


class MorphFusionNormalBgResidualChannels:
    """方案A: original_gray + weak_residual(alpha=0.1) + normal_bg_deviation

    Replaces gray_contrast with normal_bg_deviation as the third channel.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()
        self.bg_deviation = NormalBgDeviationChannels()

    def set_bg_stats(self, median_np, mad_np):
        self.bg_deviation.set_stats(median_np, mad_np)

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

        deviation_3ch = self.bg_deviation(image)
        deviation = deviation_3ch[0:1, :, :]

        return torch.cat([
            gray,           # channel 1: original_gray
            weak_residual,  # channel 2: weak_residual
            deviation,      # channel 3: normal_bg_deviation
        ], dim=0)


class MorphFusionContrastResidualNormalBgChannels:
    """方案B: gray_contrast + weak_residual(alpha=0.1) + normal_bg_deviation

    Keeps gray_contrast, replaces original_gray with normal_bg_deviation.
    """
    def __init__(self, alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        self.bg_deviation = NormalBgDeviationChannels()

    def set_bg_stats(self, median_np, mad_np):
        self.bg_deviation.set_stats(median_np, mad_np)

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

        deviation_3ch = self.bg_deviation(image)
        deviation = deviation_3ch[0:1, :, :]

        return torch.cat([
            gray_contrast,  # channel 1: gray_contrast
            weak_residual,  # channel 2: weak_residual
            deviation,      # channel 3: normal_bg_deviation
        ], dim=0)


class MorphFusionGaborTextureChannels:
    def __init__(self, alpha=0.2, clahe_clip_limit=2.0, clahe_tile_grid_size=8,
                 gabor_kernel_size=31, gabor_sigma=6.0, gabor_lambda=18.0, gabor_gamma=0.5,
                 texture_kernel_size=15, percentile=0.995):
        self.alpha = alpha
        self.percentile = percentile
        self.texture_kernel_size = texture_kernel_size
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        self.gabor_kernels = [
            cv2.getGaborKernel(
                (gabor_kernel_size, gabor_kernel_size),
                gabor_sigma,
                theta,
                gabor_lambda,
                gabor_gamma,
                0,
                ktype=cv2.CV_32F,
            )
            for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        ]

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def _local_std(self, gray):
        pad = self.texture_kernel_size // 2
        gray_b = gray.unsqueeze(0)
        mean = F.avg_pool2d(gray_b, self.texture_kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(gray_b * gray_b, self.texture_kernel_size, stride=1, padding=pad)
        var = (mean_sq - mean * mean).clamp_min(0.0)
        return torch.sqrt(var).squeeze(0)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)

        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        gray_contrast = torch.from_numpy(gray_contrast).unsqueeze(0)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        gabor_responses = [
            np.abs(cv2.filter2D(gray_np, cv2.CV_32F, kernel))
            for kernel in self.gabor_kernels
        ]
        gabor_broad = np.max(np.stack(gabor_responses, axis=0), axis=0)
        gabor_broad = self._robust_normalize(torch.from_numpy(gabor_broad).unsqueeze(0))

        local_std = self._robust_normalize(self._local_std(gray))
        texture_mix = torch.maximum(gabor_broad, local_std)

        return torch.cat([
            gray_contrast,
            weak_residual,
            texture_mix,
        ], dim=0)


class MorphFusionGrayResEnergyChannels:
    def __init__(self, alpha=0.2, kernel_size=15):
        self.alpha = alpha
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
        weak_residual = self._minmax(gray + self.alpha * residual)

        pad = self.kernel_size // 2
        local_sq_mean = F.avg_pool2d((gray * gray).unsqueeze(0), self.kernel_size, stride=1, padding=pad).squeeze(0)
        local_energy = torch.sqrt(local_sq_mean.clamp_min(0.0))

        return torch.cat([
            gray,
            weak_residual,
            self._minmax(local_energy),
        ], dim=0)


class MorphFusionGrayResBandChannels:
    def __init__(self, alpha=0.2, kernel_size=15, smooth_mix=0.5, profile_mix=0.5):
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.smooth_mix = smooth_mix
        self.profile_mix = profile_mix
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

        pad = self.kernel_size // 2
        local_smooth = F.avg_pool2d(gray.unsqueeze(0), self.kernel_size, stride=1, padding=pad).squeeze(0)
        row_band_profile = local_smooth.mean(dim=-1, keepdim=True).expand_as(gray)
        band_response = self.smooth_mix * local_smooth + self.profile_mix * row_band_profile

        return torch.cat([
            gray,
            weak_residual,
            self._minmax(band_response),
        ], dim=0)


class ChirpDirectionalChannels:
    def __init__(self, sigma=1.0, percentile=0.995):
        self.sigma = sigma
        self.percentile = percentile
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

        kernel_45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32)
        kernel_135 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
        resp_45 = np.abs(cv2.filter2D(smoothed, cv2.CV_32F, kernel_45))
        resp_135 = np.abs(cv2.filter2D(smoothed, cv2.CV_32F, kernel_135))
        diagonal_resp = np.maximum(resp_45, resp_135)
        diagonal_resp = gaussian_filter(diagonal_resp, sigma=1.0)

        grad_mag = torch.from_numpy(grad_mag).unsqueeze(0)
        diagonal_resp = torch.from_numpy(diagonal_resp).unsqueeze(0)

        return torch.cat([
            gray,
            self._robust_normalize(grad_mag),
            self._robust_normalize(diagonal_resp),
        ], dim=0)


class ChirpRidgeChannels:
    def __init__(self, sigma=1.0, percentile=0.995):
        self.sigma = sigma
        self.percentile = percentile
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        kernel_45 = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, 3, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.float32)
        kernel_135 = np.fliplr(kernel_45)
        kernel_45 = kernel_45 / kernel_45.sum()
        kernel_135 = kernel_135 / kernel_135.sum()

        ridge_45 = cv2.filter2D(smoothed, cv2.CV_32F, kernel_45)
        ridge_135 = cv2.filter2D(smoothed, cv2.CV_32F, kernel_135)
        ridge_resp = np.maximum(ridge_45, ridge_135)
        ridge_resp = np.maximum(ridge_resp - smoothed, 0.0)

        continuity_45 = cv2.filter2D(ridge_resp, cv2.CV_32F, kernel_45)
        continuity_135 = cv2.filter2D(ridge_resp, cv2.CV_32F, kernel_135)
        continuity = np.maximum(continuity_45, continuity_135)

        ridge_resp = torch.from_numpy(ridge_resp).unsqueeze(0)
        continuity = torch.from_numpy(continuity).unsqueeze(0)

        return torch.cat([
            gray,
            self._robust_normalize(ridge_resp),
            self._robust_normalize(continuity),
        ], dim=0)


class ChirpTrackEnhanceChannels:
    def __init__(self, img_resize=240, img_cropsize=240, sigma_small=0.8, sigma_large=3.0, percentile=0.995, kernel_size=11, angles=(-60, -45, -30, 30, 45, 60)):
        self.img_resize = img_resize
        self.img_cropsize = img_cropsize
        self.sigma_small = sigma_small
        self.sigma_large = sigma_large
        self.percentile = percentile
        self.kernel_size = kernel_size
        self.angles = angles
        self.to_tensor = transforms.ToTensor()
        self.kernels = [self._make_line_kernel(kernel_size, angle) for angle in angles]

    @staticmethod
    def _make_line_kernel(kernel_size, angle_deg):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        radius = kernel_size // 2
        theta = np.deg2rad(angle_deg)
        dx = int(round(radius * np.cos(theta)))
        dy = int(round(radius * np.sin(theta)))
        pt1 = (center - dx, center - dy)
        pt2 = (center + dx, center + dy)
        cv2.line(kernel, pt1, pt2, color=1, thickness=1)
        kernel = kernel.astype(np.float32)
        s = kernel.sum()
        if s > 0:
            kernel /= s
        return kernel

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    def __call__(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy().astype(np.float32)

        fine = gaussian_filter(gray_np, sigma=self.sigma_small)
        coarse = gaussian_filter(gray_np, sigma=self.sigma_large)
        local_bright = np.maximum(fine - coarse, 0.0)

        track_resp = []
        continuity_resp = []
        for kernel in self.kernels:
            track = cv2.filter2D(local_bright, cv2.CV_32F, kernel)
            continuity = cv2.filter2D(track, cv2.CV_32F, kernel)
            track_resp.append(track)
            continuity_resp.append(continuity)

        track_map = np.max(np.stack(track_resp, axis=0), axis=0)
        continuity_map = np.max(np.stack(continuity_resp, axis=0), axis=0)
        continuity_map = np.maximum(continuity_map - 0.25 * track_map, 0.0)

        feature = torch.cat([
            gray,
            self._robust_normalize(torch.from_numpy(track_map).unsqueeze(0)),
            self._robust_normalize(torch.from_numpy(continuity_map).unsqueeze(0)),
        ], dim=0)

        feature = TF.resize(feature, [self.img_resize, self.img_resize], interpolation=transforms.InterpolationMode.BICUBIC)
        feature = TF.center_crop(feature, [self.img_cropsize, self.img_cropsize])
        return feature


class ChirpRGBTrackChannels:
    def __init__(self, sigma_small=0.8, sigma_large=2.5, percentile=0.995, kernel_size=9, alpha=0.35, angles=(-60, -45, -30, 30, 45, 60)):
        self.sigma_small = sigma_small
        self.sigma_large = sigma_large
        self.percentile = percentile
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.angles = angles
        self.to_tensor = transforms.ToTensor()
        self.kernels = [self._make_line_kernel(kernel_size, angle) for angle in angles]

    @staticmethod
    def _make_line_kernel(kernel_size, angle_deg):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        radius = kernel_size // 2
        theta = np.deg2rad(angle_deg)
        dx = int(round(radius * np.cos(theta)))
        dy = int(round(radius * np.sin(theta)))
        pt1 = (center - dx, center - dy)
        pt2 = (center + dx, center + dy)
        cv2.line(kernel, pt1, pt2, color=1, thickness=1)
        kernel = kernel.astype(np.float32)
        s = kernel.sum()
        if s > 0:
            kernel /= s
        return kernel

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    def __call__(self, image):
        rgb = self.to_tensor(image.convert('RGB'))
        gray = self.to_tensor(image.convert('L')).squeeze(0).numpy().astype(np.float32)

        fine = gaussian_filter(gray, sigma=self.sigma_small)
        coarse = gaussian_filter(gray, sigma=self.sigma_large)
        local_bright = np.maximum(fine - coarse, 0.0)

        track_resp = []
        continuity_resp = []
        for kernel in self.kernels:
            track = cv2.filter2D(local_bright, cv2.CV_32F, kernel)
            continuity = cv2.filter2D(track, cv2.CV_32F, kernel)
            track_resp.append(track)
            continuity_resp.append(continuity)

        track_map = np.max(np.stack(track_resp, axis=0), axis=0)
        continuity_map = np.max(np.stack(continuity_resp, axis=0), axis=0)
        enhance = np.maximum(track_map, continuity_map)
        enhance = self._robust_normalize(torch.from_numpy(enhance).unsqueeze(0))

        return torch.clamp(rgb + self.alpha * enhance.repeat(3, 1, 1), 0.0, 1.0)


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


class DSSSGrayWeakResidualResidualChannels:
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
        return torch.cat([gray, weak_residual, weak_residual], dim=0)


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


class VisualClassPromptAdapter(nn.Module):
    def __init__(self, visual_dim, text_dim, token_num=2, bottleneck_ratio=0.25, alpha=0.2):
        super().__init__()
        self.token_num = int(token_num)
        self.text_dim = int(text_dim)
        self.alpha = float(alpha)
        hidden_dim = max(1, int(visual_dim * bottleneck_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_num * self.text_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prototype, dtype=None):
        if prototype.ndim == 1:
            prototype = prototype.unsqueeze(0)
        tokens = self.net(prototype.float()).reshape(prototype.shape[0], self.token_num, self.text_dim)
        tokens = self.alpha * tokens
        if dtype is not None:
            tokens = tokens.to(dtype)
        return tokens


class LearnableScoreFusionHead(nn.Module):
    """A tiny residual head that predicts a correction to the text score."""

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


class TinyMambaFusionBlock(nn.Module):
    """Lightweight Mamba-inspired token mixer implemented locally."""

    def __init__(self, dim, expansion=2, kernel_size=3, dropout=0.0):
        super().__init__()
        inner_dim = max(1, int(dim * expansion))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.dwconv = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=kernel_size // 2, groups=inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        u, v = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.gelu(u)
        x = self.out_proj(self.dropout(u * torch.sigmoid(v)))
        return residual + x


class TinyCNNPatchEncoder(nn.Module):
    def __init__(self, dim, grid_size):
        super().__init__()
        hidden1 = max(32, dim // 2)
        hidden2 = max(32, dim)
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
        self.pool = nn.AdaptiveAvgPool2d(grid_size)

    def forward(self, image):
        feat = self.stem(image.float())
        tokens = self.pool(feat).flatten(2).transpose(1, 2)
        return tokens


class CNNMambaLocalBranch(nn.Module):
    def __init__(self, dim, grid_size, num_blocks=1, dropout=0.0):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.cnn_encoder = TinyCNNPatchEncoder(dim=dim, grid_size=self.grid_size)
        self.mamba_blocks = nn.ModuleList([
            TinyMambaFusionBlock(dim=dim, dropout=dropout)
            for _ in range(max(1, int(num_blocks)))
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, image):
        tokens = self.cnn_encoder(image)
        for block in self.mamba_blocks:
            tokens = block(tokens)
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
        return self.visual_class_prompt_adapter(prototype, dtype=suffix.dtype)

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
        self.visual_score_alpha = kwargs.get('visual_score_alpha', 1.0)
        self.visual_score_beta = kwargs.get('visual_score_beta', 1.0)
        self.visual_score_gamma = kwargs.get('visual_score_gamma', 0.0)
        self.visual_freq_position_weight = kwargs.get('visual_freq_position_weight', 0.0)
        self.normal_dist_ridge = kwargs.get('normal_dist_ridge', 1e-4)
        self.use_learnable_score_fusion = kwargs.get('learnable_score_fusion', False)
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
        elif self.input_mode == 'spectral_gradient_v2':
            self.transform = transforms.Compose(pre_resize_crop + [
                SpectrogramGradientChannelsV2(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_plus':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionPlusChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_dualgrad':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionDualGradChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_balanced':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionBalancedChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_resgrad':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResGradChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gabor_residual':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGaborResidualChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gabor_residual_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGaborResidualChannels(alpha=0.1),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gabor_texture':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGaborTextureChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gabor_directional_residual':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGaborDirectionalResidualChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_clahe10':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=1.0),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_clahe15':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=1.5),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_clahe30':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=3.0),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_tile4':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=4),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_tile16':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=16),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_clahe05':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=0.5),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_clahe80':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=8.0),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_clahe200':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=20.0),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_c5t2':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=5.0, clahe_tile_grid_size=2),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_tile2':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=2),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_a01_tile32':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualChannels(alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=32),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_residual_no_contrast_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResidualNoContrastChannels(alpha=0.1),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'gray_contrast_only':
            self.transform = transforms.Compose(pre_resize_crop + [
                GrayContrastOnlyChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_resenergy':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResEnergyChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_gray_resband':
            self.transform = transforms.Compose(pre_resize_crop + [
                MorphFusionGrayResBandChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'chirp_directional':
            self.transform = transforms.Compose(pre_resize_crop + [
                ChirpDirectionalChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'chirp_ridge':
            self.transform = transforms.Compose(pre_resize_crop + [
                ChirpRidgeChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'chirp_track_enhance':
            self.transform = transforms.Compose([
                ChirpTrackEnhanceChannels(
                    img_resize=kwargs['img_resize'],
                    img_cropsize=kwargs['img_cropsize']),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'chirp_rgb_track':
            self.transform = transforms.Compose(pre_resize_crop + [
                ChirpRGBTrackChannels(),
                transforms.Normalize(mean=mean_train, std=std_train)])
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
        elif self.input_mode == 'dsss_gray_weakresidual_weakresidual':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSGrayWeakResidualResidualChannels(alpha=0.2),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_weak_residual_only':
            self.transform = transforms.Compose(pre_resize_crop + [
                WeakResidualOnlyChannels(alpha=0.2),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_weak_residual_only_a01':
            self.transform = transforms.Compose(pre_resize_crop + [
                WeakResidualOnlyChannels(alpha=0.1),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'dsss_clahe':
            self.transform = transforms.Compose(pre_resize_crop + [
                DSSSCLAHEChannels(),
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_normal_bg_residual':
            self._normal_bg_transform = MorphFusionNormalBgResidualChannels(alpha=0.1)
            self.transform = transforms.Compose(pre_resize_crop + [
                self._normal_bg_transform,
                transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train)])
        elif self.input_mode == 'morph_fusion_contrast_residual_normal_bg':
            self._normal_bg_transform = MorphFusionContrastResidualNormalBgChannels(alpha=0.1)
            self.transform = transforms.Compose(pre_resize_crop + [
                self._normal_bg_transform,
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
                if dataset_name in {'spectrum', 'sample', 'deceptive_signal', 'rf_spe_png'}:
                    return 'spectral_gradient'
                return 'rgb'
            if input_mode == 'signal_adaptive_v2':
                if dataset_name in {'burst_signal', 'chirp_signal'}:
                    return 'spectral_gradient'
                if dataset_name == 'dsss_signal':
                    return 'dsss_weak_residual'
                if dataset_name == 'wideband_pulse':
                    return 'rgb'
                if dataset_name in {'spectrum', 'sample', 'deceptive_signal', 'rf_spe_png'}:
                    return 'spectral_gradient'
                return 'rgb'
            return input_mode
        if dataset_name in {'spectrum', 'sample', 'deceptive_signal', 'burst_signal', 'dsss_signal', 'chirp_signal', 'wideband_pulse', 'rf_spe_png'}:
            return 'spectral_gradient'
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
        if self.visual_class_prompt_adapter is None:
            return
        prototype = F.normalize(global_features.float(), dim=-1).mean(dim=0, keepdim=True)
        self.prompt_learner.set_visual_class_prompt_prototype(prototype.detach())

    def trainable_parameters(self):
        params = list(self.prompt_learner.parameters())
        if self.visual_adapters is not None:
            params.extend(self.visual_adapters.parameters())
        if self.visual_class_prompt_adapter is not None:
            params.extend(self.visual_class_prompt_adapter.parameters())
        if self.score_fusion_head is not None:
            params.extend(self.score_fusion_head.parameters())
        if self.cnn_mamba_local_branch is not None:
            params.extend(self.cnn_mamba_local_branch.parameters())
        if self.use_visual_lora:
            params.extend(
                p for name, p in self.model.visual.named_parameters()
                if 'lora_' in name and p.requires_grad
            )
        return params

    def build_learnable_fusion_features(self, textual_anomaly, visual_anomaly_map):
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
        base_logit = torch.logit(textual_anomaly.clamp(1e-4, 1.0 - 1e-4))
        return base_logit + float(self.learnable_score_fusion_alpha) * delta

    def encode_rn50_image(self, image):
        if self.rn50_model is None:
            raise RuntimeError('encode_rn50_image requires --rn50-visual-fusion True')
        image = F.interpolate(image.float(), size=(224, 224), mode='bilinear', align_corners=False)
        if self.precision == 'fp16':
            image = image.half()
        with torch.no_grad():
            rn50_features = self.rn50_model.encode_image(image)
        if isinstance(rn50_features, (list, tuple)):
            if len(rn50_features) == 1 and rn50_features[0].dim() == 2:
                rn50_features = rn50_features[0]
            elif rn50_features and rn50_features[0].dim() == 1:
                rn50_features = torch.stack(list(rn50_features), dim=0)
            else:
                rn50_features = rn50_features[0]
        return F.normalize(rn50_features, dim=-1)

    def encode_rn50_local(self, image):
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
        gallery = F.normalize(rn50_features, dim=-1)
        if self.rn50_gallery.shape[0] != gallery.shape[0] or self.rn50_gallery.shape[1] != gallery.shape[1]:
            self.rn50_gallery = gallery.new_zeros(gallery.shape)
        self.rn50_gallery.copy_(gallery.to(self.rn50_gallery.dtype))
        if rn50_local_features is not None:
            b, n, d = rn50_local_features.shape
            local_gallery = F.normalize(rn50_local_features.reshape(-1, d), dim=-1)
            if (self.rn50_local_gallery.shape[0] != local_gallery.shape[0]
                    or self.rn50_local_gallery.shape[1] != local_gallery.shape[1]):
                self.rn50_local_gallery = local_gallery.new_zeros(local_gallery.shape)
            self.rn50_local_gallery.copy_(local_gallery.to(self.rn50_local_gallery.dtype))

    def calculate_rn50_global_score(self, image):
        rn50_features = self.encode_rn50_image(image)
        score, _ = (1.0 - rn50_features @ self.rn50_gallery.t()).min(dim=-1)
        return score / 2.0

    def calculate_rn50_local_map(self, image):
        local_features = self.encode_rn50_local(image)
        score, _ = (1.0 - local_features @ self.rn50_local_gallery.t()).min(dim=-1)
        score = score / 2.0
        side = int(round(score.shape[1] ** 0.5))
        return score.reshape((image.shape[0], side, side)).unsqueeze(1)

    def calculate_rn50_local_score(self, image):
        local_map = self.calculate_rn50_local_map(image)
        flat = local_map.flatten(1)
        topk = max(1, int(flat.shape[1] * float(self.rn50_local_topk_ratio)))
        return torch.topk(flat, k=topk, dim=1).values.mean(dim=1)

    def calculate_rn50_guided_vit_score(self, image, visual_anomaly_map):
        rn50_map = self.calculate_rn50_local_map(image).to(
            device=visual_anomaly_map.device,
            dtype=visual_anomaly_map.dtype,
        )
        rn50_map = F.interpolate(
            rn50_map,
            size=visual_anomaly_map.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        weights = rn50_map.flatten(1).float()
        weights = weights - weights.amin(dim=1, keepdim=True)
        scale = weights.amax(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / scale
        temperature = max(1e-4, float(self.rn50_guidance_temperature))
        weights = torch.softmax(weights / temperature, dim=1).to(visual_anomaly_map.dtype)
        vit_scores = visual_anomaly_map.flatten(1)
        return (weights * vit_scores).sum(dim=1)

    def calculate_rn50_visual_score(self, image):
        if self.rn50_score_mode == 'global':
            return self.calculate_rn50_global_score(image)
        if self.rn50_score_mode == 'local_topk':
            return self.calculate_rn50_local_score(image)
        if self.rn50_score_mode == 'both':
            return 0.5 * (self.calculate_rn50_global_score(image) + self.calculate_rn50_local_score(image))
        raise ValueError(f'Unknown rn50_score_mode: {self.rn50_score_mode}')

    def encode_cnn_mamba_local(self, image):
        if self.cnn_mamba_local_branch is None:
            raise RuntimeError('encode_cnn_mamba_local requires --cnn-vit-mamba-fusion True')
        return self.cnn_mamba_local_branch(image)

    def build_cnn_mamba_gallery(self, cnn_mamba_features):
        b, n, d = cnn_mamba_features.shape
        gallery = F.normalize(cnn_mamba_features.reshape(-1, d), dim=-1)
        if self.cnn_mamba_gallery.shape[0] != gallery.shape[0] or self.cnn_mamba_gallery.shape[1] != gallery.shape[1]:
            self.cnn_mamba_gallery = gallery.new_zeros(gallery.shape)
        self.cnn_mamba_gallery.copy_(gallery.to(self.cnn_mamba_gallery.dtype))

        global_gallery = F.normalize(cnn_mamba_features.mean(dim=1), dim=-1)
        if (self.cnn_mamba_global_gallery.shape[0] != global_gallery.shape[0]
                or self.cnn_mamba_global_gallery.shape[1] != global_gallery.shape[1]):
            self.cnn_mamba_global_gallery = global_gallery.new_zeros(global_gallery.shape)
        self.cnn_mamba_global_gallery.copy_(global_gallery.to(self.cnn_mamba_global_gallery.dtype))

    def aggregate_cnn_mamba_by_anomaly(self, local_features, patch_score):
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
        local_features = self.encode_cnn_mamba_local(image)
        score, _ = (1.0 - local_features @ self.cnn_mamba_gallery.t()).min(dim=-1)
        score = score / 2.0
        patch_map = score.reshape((image.shape[0], self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        pooled_feature, _ = self.aggregate_cnn_mamba_by_anomaly(local_features, score)
        image_score, _ = (1.0 - pooled_feature @ self.cnn_mamba_global_gallery.t()).min(dim=-1)
        image_score = image_score / 2.0
        return image_score, patch_map

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

    def calculate_normal_distribution_score(self, visual_features, mode):
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

    def fuse_cls_scores(self, textual_anomaly, visual_anomaly_map, visual_features=None):
        if self.cls_score_mode == 'text_only':
            return textual_anomaly

        if self.cls_score_mode in {'normal_center', 'normal_mahalanobis', 'text_normal_center', 'text_normal_mahalanobis'}:
            if visual_features is None:
                return textual_anomaly
            dist_mode = 'center' if 'center' in self.cls_score_mode else 'mahalanobis'
            normal_score = self.calculate_normal_distribution_score(visual_features, dist_mode)
            if self.cls_score_mode.startswith('text_'):
                return self.visual_score_alpha * textual_anomaly + self.visual_score_beta * normal_score
            return normal_score

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

    def score_cached(self, visual_features, task='cls', return_raw_map=False, image=None):
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
        image_scores = np.asarray(textual_anomaly, dtype=np.float32)
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
        elif self.rn50_model is None:
            image_scores = self.fuse_cls_scores(textual_anomaly, visual_anomaly_map, visual_features)
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
