"""Time-frequency attention adversarial autoencoder for spectrum images.

This module implements the spectrogram reconstruction part of TFAM-AAE.  The
paper's separate ``U_k`` branch consumes raw IQ samples and is intentionally
not represented here because the current RF protocol stores dBm spectrograms,
not IQ recordings.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _downsample_size(size: int) -> int:
    """Return the spatial size after one stride-two encoder convolution."""

    return max(1, (int(size) + 2 - 4) // 2 + 1)


class TimeFrequencyAttention(nn.Module):
    """A compact TAM/FAM block following the paper's Eq. (5)-(6).

    The descriptors collapse the orthogonal axis and channel dimension.  The
    two branches then produce one time weight and one frequency weight, which
    are broadcast back over the feature map before a refinement convolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.time_branch = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(f"Expected BCHW features, got {tuple(features.shape)}")

        # Pool over channels and the orthogonal axis to form the TAM/FAM
        # descriptors.  Keeping max and mean mirrors the paper's two pooling
        # paths while remaining valid for arbitrary feature-channel counts.
        time_mean = features.mean(dim=(1, 3), keepdim=True)
        time_max = features.amax(dim=(1, 3), keepdim=True)
        time_descriptor = torch.cat((time_mean, time_max), dim=1)
        time_weight = self.time_branch(time_descriptor)

        frequency_mean = features.mean(dim=(1, 2), keepdim=True)
        frequency_max = features.amax(dim=(1, 2), keepdim=True)
        frequency_descriptor = torch.cat((frequency_mean, frequency_max), dim=1)
        frequency_weight = self.frequency_branch(frequency_descriptor)

        combined = time_weight.expand(-1, -1, -1, features.shape[-1])
        combined = combined + frequency_weight.expand(
            -1, -1, features.shape[-2], -1
        )
        refined = self.refine(features * (1.0 + 0.5 * combined))
        return refined, time_weight, frequency_weight


class TFAMEncoder(nn.Module):
    """Convolutional encoder with a TFAM block after each downsampling stage."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        base_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        height, width = (int(input_shape[0]), int(input_shape[1]))
        if height < 8 or width < 8:
            raise ValueError("TFAM input dimensions must be at least 8x8")

        channels = (int(base_channels), int(base_channels) * 2, int(base_channels) * 4)
        self.input_shape = (height, width)
        self.channels = channels
        self.convolutions = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = 1
        spatial_shapes = []
        for out_channels in channels:
            self.convolutions.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.attentions.append(TimeFrequencyAttention(out_channels))
            height = _downsample_size(height)
            width = _downsample_size(width)
            spatial_shapes.append((out_channels, height, width))
            in_channels = out_channels

        self.feature_shapes = tuple(spatial_shapes)
        deepest = self.feature_shapes[-1]
        self.to_latent = nn.Linear(
            deepest[0] * deepest[1] * deepest[2], int(latent_dim)
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        features = inputs
        attention_maps = []
        for convolution, attention in zip(self.convolutions, self.attentions):
            features = F.gelu(convolution(features))
            features, time_weight, frequency_weight = attention(features)
            attention_maps.append((time_weight, frequency_weight))
        latent = self.to_latent(features.flatten(1))
        return latent, attention_maps


class TFAMDecoder(nn.Module):
    """Decoder that applies TFAM-guided refinement during reconstruction."""

    def __init__(self, encoder: TFAMEncoder, latent_dim: int):
        super().__init__()
        self.input_shape = encoder.input_shape
        deepest = encoder.feature_shapes[-1]
        self.from_latent = nn.Linear(int(latent_dim), deepest[0] * deepest[1] * deepest[2])

        in_channels = [encoder.channels[-1], encoder.channels[-2], encoder.channels[0]]
        out_channels = [encoder.channels[-2], encoder.channels[0], encoder.channels[0]]
        self.convolutions = nn.ModuleList(
            nn.Conv2d(in_channels[index], out_channels[index], 3, padding=1)
            for index in range(len(in_channels))
        )
        self.attentions = nn.ModuleList(
            TimeFrequencyAttention(channels) for channels in out_channels
        )
        self.output = nn.Conv2d(out_channels[-1], 1, kernel_size=3, padding=1)
        self.decoder_shapes = tuple(
            shape[-2:]
            for shape in (
                encoder.feature_shapes[-2],
                encoder.feature_shapes[0],
                encoder.input_shape,
            )
        )
        self.deepest_shape = deepest[-2:]

    def forward(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = latent.shape[0]
        features = self.from_latent(latent).reshape(
            batch_size,
            -1,
            self.deepest_shape[0],
            self.deepest_shape[1],
        )
        attention_maps = []
        for convolution, attention, target_shape in zip(
            self.convolutions, self.attentions, self.decoder_shapes
        ):
            features = F.interpolate(
                features,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
            features = F.gelu(convolution(features))
            features, time_weight, frequency_weight = attention(features)
            attention_maps.append((time_weight, frequency_weight))
        reconstruction = torch.sigmoid(self.output(features))
        return reconstruction, attention_maps


class TFAMLatentDiscriminator(nn.Module):
    """Discriminator used by the adversarial autoencoder latent prior."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.layers(latent)


class TFAMAAE(nn.Module):
    """TFAM-AAE spectrogram reconstruction model.

    Inputs and outputs are single-channel tensors normalized to ``[0, 1]``.
    The model is intentionally fixed to one input shape per training run so
    that the latent projection remains deterministic.
    """

    def __init__(
        self,
        input_shape: tuple[int, int] = (64, 64),
        base_channels: int = 16,
        latent_dim: int = 64,
        discriminator_hidden_dim: int = 128,
        discriminator_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_shape = (int(input_shape[0]), int(input_shape[1]))
        self.encoder = TFAMEncoder(self.input_shape, base_channels, latent_dim)
        self.decoder = TFAMDecoder(self.encoder, latent_dim)
        self.discriminator = TFAMLatentDiscriminator(
            latent_dim,
            discriminator_hidden_dim,
            discriminator_dropout,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)[0]

    def reconstruct(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, _encoder_attention = self.encoder(inputs)
        reconstruction, _decoder_attention = self.decoder(latent)
        return reconstruction, latent

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(inputs)[0]


def reconstruction_mse(
    inputs: torch.Tensor, reconstruction: torch.Tensor
) -> torch.Tensor:
    """Return one TFAM Eq. (9) mean-square score per input sample."""

    if inputs.shape != reconstruction.shape:
        raise ValueError(
            "Input and reconstruction shapes differ: "
            f"{tuple(inputs.shape)} vs {tuple(reconstruction.shape)}"
        )
    return (inputs - reconstruction).square().flatten(1).mean(dim=1)
