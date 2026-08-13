"""Paper-driven core implementation of UDMA for spectrum anomaly detection.

This module follows Qi et al., *Unsupervised Spectrum Anomaly Detection With
Distillation and Memory Enhanced Autoencoders*, IEEE Internet of Things
Journal, 2024, DOI: 10.1109/JIOT.2024.3424837.  It implements the three-network
design, equations (2)--(27), and the paper's staged training interface.

The paper does not publish source code and leaves details needed for an exact
reproduction unspecified.  The following are explicit implementation choices:

* The pretrained network ``P`` and the dense projection ``D`` in equation (2)
  are not identified.  ``teacher_distillation_loss`` therefore accepts frozen
  reference features from any external extractor, globally pools feature maps,
  and uses a learned linear projection.  This avoids an input-size-dependent
  flatten operation.
* Table II fixes the student dense layers to an 8x64 input, while the paper's
  OTA experiment uses 64x256 inputs.  Here the students are fully convolutional
  and each latent spatial vector is a memory query.  Memory retrieval is fused
  by a 1x1 convolution.  This preserves localization and supports variable
  time-frequency sizes without lazy or fixed flatten dimensions.
* Padding, nonlinearities, decoder upsampling, and the pretrained model are not
  specified.  We use same-padded 3x3 convolutions, ReLU, max pooling, and
  bilinear interpolation.  The teacher still has the stated three 3x3 layers
  and a 7x7 receptive field.
* The default channels are deliberately lighter than Tables I--II for safe CPU
  testing.  Paper-scale teacher widths can be requested with
  ``feature_channels=128`` and ``teacher_hidden=(128, 256)``.
* The hard-shrink thresholds, memory update rate, and separateness margin are
  not fixed by the paper.  Addressing defaults to 1/M as suggested by the
  paper; updating defaults to 1/N_query because equation (14) normalizes over
  queries.  Both are exposed as constructor arguments.  If hard shrink removes
  every weight, the closest item/query is used as a finite one-hot fallback.
* Equation (22) defines memory loss but the paper does not explicitly state how
  it is combined with equation (9).  ``student_training_loss`` adds them.
* Teacher statistics are frozen from normal training data.  The paper's phrase
  "anomaly free samples in the testset" is not used because it would require
  test-set normal labels at inference time.

This is therefore an auditable paper-based reimplementation, not a claim of an
official bit-for-bit reproduction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _check_spectrogram(x: Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
    if min(x.shape[-2:]) < 8:
        raise ValueError("UDMA students require both spatial dimensions >= 8")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=False),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UDMATeacher(nn.Module):
    """Three-layer CNN teacher with the paper's 7x7 receptive field."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        hidden_channels: Sequence[int] = (32, 64),
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("hidden_channels must contain exactly two widths")
        h1, h2 = (int(width) for width in hidden_channels)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, h1, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(h1, h2, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(h2, feature_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _check_spectrogram(x)
        return self.network(x)


class _StudentBackbone(nn.Module):
    """Size-independent convolutional encoder/decoder shared in design."""

    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        encoder_channels: Sequence[int],
        latent_channels: int,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 3:
            raise ValueError("encoder_channels must contain exactly three widths")
        c1, c2, c3 = (int(width) for width in encoder_channels)
        self.encoder_blocks = nn.ModuleList(
            [
                _ConvBlock(in_channels, c1),
                _ConvBlock(c1, c2),
                _ConvBlock(c2, c3),
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _ConvBlock(c3, latent_channels, depth=1)
        self.decoder_blocks = nn.ModuleList(
            [
                _ConvBlock(latent_channels, c3, depth=1),
                _ConvBlock(c3, c2, depth=1),
                _ConvBlock(c2, feature_channels, depth=1),
            ]
        )

    def encode(self, x: Tensor) -> tuple[Tensor, list[tuple[int, int]]]:
        _check_spectrogram(x)
        sizes: list[tuple[int, int]] = []
        for block in self.encoder_blocks:
            x = block(x)
            sizes.append((x.shape[-2], x.shape[-1]))
            x = self.pool(x)
        return self.bottleneck(x), sizes

    def decode(self, z: Tensor, sizes: Sequence[tuple[int, int]]) -> Tensor:
        x = z
        for block, target_size in zip(self.decoder_blocks, reversed(sizes), strict=True):
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            x = block(x)
        return x


class UDMAAEStudent(_StudentBackbone):
    """Plain autoencoder student from equations (6) and (8)."""

    def forward(self, x: Tensor) -> Tensor:
        z, sizes = self.encode(x)
        return self.decode(z, sizes)


class UDMAMemory(nn.Module):
    """Cosine-addressed normal-pattern memory from equations (10)--(22)."""

    def __init__(
        self,
        memory_size: int,
        feature_dim: int,
        shrink_threshold: float | None = None,
        update_threshold: float | None = None,
        update_rate: float = 0.1,
        separateness_margin: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if memory_size < 2:
            raise ValueError("memory_size must be at least 2")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.memory_size = int(memory_size)
        self.feature_dim = int(feature_dim)
        self.shrink_threshold = float(
            1.0 / memory_size if shrink_threshold is None else shrink_threshold
        )
        self.update_threshold = (
            None if update_threshold is None else float(update_threshold)
        )
        self.update_rate = float(update_rate)
        self.separateness_margin = float(separateness_margin)
        self.eps = float(eps)
        initial = F.normalize(torch.randn(memory_size, feature_dim), dim=1, eps=eps)
        # Equation (16) explicitly updates memory; it is state, not an optimizer
        # parameter. Encoder gradients still flow through reads from this buffer.
        self.register_buffer("items", initial)

    def _hard_shrink(self, weights: Tensor, threshold: float, dim: int) -> Tensor:
        shrunk = F.relu(weights - threshold) * weights
        shrunk = shrunk / (torch.abs(weights - threshold) + self.eps)
        normalizer = shrunk.sum(dim=dim, keepdim=True)

        # The paper does not define the all-zero case. Preserve the closest item
        # with a one-hot fallback so addressing and memory updates remain finite.
        winner = weights.argmax(dim=dim, keepdim=True)
        fallback = torch.zeros_like(weights).scatter_(dim, winner, 1.0)
        return torch.where(normalizer > self.eps, shrunk / normalizer.clamp_min(self.eps), fallback)

    def similarities(self, queries: Tensor) -> Tensor:
        if queries.ndim != 2 or queries.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected queries [N, {self.feature_dim}], got {tuple(queries.shape)}"
            )
        query_unit = F.normalize(queries, dim=1, eps=self.eps)
        memory_unit = F.normalize(self.items, dim=1, eps=self.eps)
        return query_unit @ memory_unit.transpose(0, 1)

    def address(self, queries: Tensor) -> dict[str, Tensor]:
        similarity = self.similarities(queries)
        dense_weights = torch.softmax(similarity, dim=1)
        weights = self._hard_shrink(dense_weights, self.shrink_threshold, dim=1)
        retrieved = weights @ self.items

        top_indices = dense_weights.topk(k=2, dim=1).indices
        positive = self.items[top_indices[:, 0]]
        negative = self.items[top_indices[:, 1]]
        positive_distance = (queries - positive).square().sum(dim=1)
        negative_distance = (queries - negative).square().sum(dim=1)
        compactness = positive_distance.mean()
        separateness = F.relu(
            self.separateness_margin + positive_distance - negative_distance
        ).mean()
        return {
            "retrieved": retrieved,
            "weights": weights,
            "dense_weights": dense_weights,
            "similarity": similarity,
            "compactness_loss": compactness,
            "separateness_loss": separateness,
        }

    @torch.no_grad()
    def update(self, queries: Tensor) -> None:
        """Apply equations (14)--(17); call after optimizer.step()."""

        similarity = self.similarities(queries.detach())
        update_weights = torch.softmax(similarity, dim=0)
        update_threshold = (
            1.0 / queries.shape[0]
            if self.update_threshold is None
            else self.update_threshold
        )
        update_weights = self._hard_shrink(update_weights, update_threshold, dim=0)
        delta = update_weights.transpose(0, 1) @ queries.detach()
        updated = self.items + self.update_rate * delta
        self.items.copy_(F.normalize(updated, dim=1, eps=self.eps))


class UDMAMemAEStudent(_StudentBackbone):
    """Memory-enhanced AE student with spatial, size-independent queries."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        encoder_channels: Sequence[int] = (16, 32, 32),
        latent_channels: int = 32,
        memory_size: int = 10,
        shrink_threshold: float | None = None,
        update_threshold: float | None = None,
        memory_update_rate: float = 0.1,
        separateness_margin: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            feature_channels=feature_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
        )
        self.memory = UDMAMemory(
            memory_size=memory_size,
            feature_dim=latent_channels,
            shrink_threshold=shrink_threshold,
            update_threshold=update_threshold,
            update_rate=memory_update_rate,
            separateness_margin=separateness_margin,
        )
        self.memory_fusion = nn.Conv2d(2 * latent_channels, latent_channels, kernel_size=1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        z, sizes = self.encode(x)
        batch, channels, height, width = z.shape
        queries = z.permute(0, 2, 3, 1).reshape(-1, channels)
        memory_output = self.memory.address(queries)
        retrieved = memory_output["retrieved"].reshape(batch, height, width, channels)
        retrieved_map = retrieved.permute(0, 3, 1, 2).contiguous()
        fused = self.memory_fusion(torch.cat([z, retrieved_map], dim=1))
        output = self.decode(fused, sizes)
        return {
            "output": output,
            "queries": queries,
            "weights": memory_output["weights"],
            "dense_weights": memory_output["dense_weights"],
            "compactness_loss": memory_output["compactness_loss"],
            "separateness_loss": memory_output["separateness_loss"],
        }


class UDMA(nn.Module):
    """UDMA teacher, AE student, MemAE student, losses, maps, and scores."""

    _VALID_PHASES = {"teacher", "students", "inference"}

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        teacher_hidden: Sequence[int] = (32, 64),
        encoder_channels: Sequence[int] = (16, 32, 32),
        latent_channels: int = 32,
        memory_size: int = 10,
        reference_dim: int | None = None,
        shrink_threshold: float | None = None,
        update_threshold: float | None = None,
        memory_update_rate: float = 0.1,
        separateness_margin: float = 1.0,
        discrepancy_weights: Sequence[float] = (0.5, 0.5, 0.5),
        compactness_weight: float = 0.1,
        separateness_weight: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(discrepancy_weights) != 3:
            raise ValueError("discrepancy_weights must contain three values")
        self.feature_channels = int(feature_channels)
        self.reference_dim = int(reference_dim or feature_channels)
        self.discrepancy_weights = tuple(float(value) for value in discrepancy_weights)
        self.compactness_weight = float(compactness_weight)
        self.separateness_weight = float(separateness_weight)
        self.eps = float(eps)

        self.teacher = UDMATeacher(
            in_channels=in_channels,
            feature_channels=feature_channels,
            hidden_channels=teacher_hidden,
        )
        self.teacher_projection = nn.Linear(feature_channels, self.reference_dim)
        student_kwargs = dict(
            in_channels=in_channels,
            feature_channels=feature_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
        )
        self.ae_student = UDMAAEStudent(**student_kwargs)
        self.memae_student = UDMAMemAEStudent(
            **student_kwargs,
            memory_size=memory_size,
            shrink_threshold=shrink_threshold,
            update_threshold=update_threshold,
            memory_update_rate=memory_update_rate,
            separateness_margin=separateness_margin,
        )

        self.register_buffer(
            "teacher_mean", torch.zeros(1, feature_channels, 1, 1)
        )
        self.register_buffer(
            "teacher_std", torch.ones(1, feature_channels, 1, 1)
        )
        self.register_buffer("teacher_statistics_fitted", torch.tensor(False))

    def configure_phase(self, phase: str) -> None:
        """Freeze irrelevant networks for teacher, student, or inference phase."""

        if phase not in self._VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(self._VALID_PHASES)}")
        teacher_trainable = phase == "teacher"
        students_trainable = phase == "students"
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(teacher_trainable)
        for parameter in self.teacher_projection.parameters():
            parameter.requires_grad_(teacher_trainable)
        for module in (self.ae_student, self.memae_student):
            for parameter in module.parameters():
                parameter.requires_grad_(students_trainable)
        self.teacher.train(teacher_trainable)
        self.teacher_projection.train(teacher_trainable)
        self.ae_student.train(students_trainable)
        self.memae_student.train(students_trainable)

    @staticmethod
    def _reference_vector(reference_features: Tensor) -> Tensor:
        if reference_features.ndim == 2:
            return reference_features
        if reference_features.ndim == 4:
            return F.adaptive_avg_pool2d(reference_features, output_size=1).flatten(1)
        raise ValueError("reference features must have shape [B, D] or [B, D, H, W]")

    def teacher_distillation_loss(
        self, x: Tensor, reference_features: Tensor
    ) -> dict[str, Tensor]:
        """Equation (2), for the teacher-distillation training phase."""

        teacher_features = self.teacher(x)
        teacher_vector = F.adaptive_avg_pool2d(teacher_features, output_size=1).flatten(1)
        projected = self.teacher_projection(teacher_vector)
        target = self._reference_vector(reference_features).detach()
        if projected.shape != target.shape:
            raise ValueError(
                "projected teacher/reference mismatch: "
                f"{tuple(projected.shape)} vs {tuple(target.shape)}"
            )
        loss = F.mse_loss(projected, target)
        return {
            "loss": loss,
            "teacher_features": teacher_features,
            "projected_teacher": projected,
            "reference_features": target,
        }

    @staticmethod
    def feature_statistics(features: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        """Compute equations (3)--(4) over batch and spatial dimensions."""

        if features.ndim != 4:
            raise ValueError("teacher features must have shape [B, C, H, W]")
        mean = features.mean(dim=(0, 2, 3), keepdim=True)
        variance = (features - mean).square().mean(dim=(0, 2, 3), keepdim=True)
        return mean, variance.sqrt().clamp_min(eps)

    @torch.no_grad()
    def fit_teacher_statistics(self, normal_x: Tensor) -> tuple[Tensor, Tensor]:
        """Fit frozen normal-training statistics for equation (5)."""

        features = self.teacher(normal_x)
        mean, std = self.feature_statistics(features, eps=self.eps)
        self.set_teacher_statistics(mean, std)
        return mean, std

    @torch.no_grad()
    def set_teacher_statistics(self, mean: Tensor, std: Tensor) -> None:
        expected = self.teacher_mean.shape
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"teacher statistics must both have shape {tuple(expected)}"
            )
        self.teacher_mean.copy_(mean)
        self.teacher_std.copy_(std.clamp_min(self.eps))
        self.teacher_statistics_fitted.fill_(True)

    def normalized_teacher(self, x: Tensor) -> Tensor:
        """Equation (5) with statistics frozen from normal training data."""

        if not bool(self.teacher_statistics_fitted.item()):
            raise RuntimeError("fit or set teacher statistics before student training/inference")
        return (self.teacher(x) - self.teacher_mean) / (self.teacher_std + self.eps)

    def student_training_loss(self, x: Tensor) -> dict[str, Tensor]:
        """Equations (6)--(9) plus memory loss (19)--(22)."""

        with torch.no_grad():
            teacher_target = self.normalized_teacher(x)
        ae_output = self.ae_student(x)
        memae = self.memae_student(x)
        memae_output = memae["output"]

        student_teacher_ae = F.mse_loss(ae_output, teacher_target)
        student_teacher_memae = F.mse_loss(memae_output, teacher_target)
        student_student = F.mse_loss(ae_output, memae_output)
        w1, w2, w3 = self.discrepancy_weights
        distillation_loss = (
            w1 * student_teacher_ae
            + w2 * student_teacher_memae
            + w3 * student_student
        )
        memory_loss = (
            self.compactness_weight * memae["compactness_loss"]
            + self.separateness_weight * memae["separateness_loss"]
        )
        total = distillation_loss + memory_loss
        return {
            "loss": total,
            "distillation_loss": distillation_loss,
            "memory_loss": memory_loss,
            "student_teacher_ae_loss": student_teacher_ae,
            "student_teacher_memae_loss": student_teacher_memae,
            "student_student_loss": student_student,
            "compactness_loss": memae["compactness_loss"],
            "separateness_loss": memae["separateness_loss"],
            "teacher_target": teacher_target,
            "ae_output": ae_output,
            "memae_output": memae_output,
            "memory_queries": memae["queries"],
            "memory_weights": memae["weights"],
        }

    @torch.no_grad()
    def update_memory(self, queries: Tensor) -> None:
        """Update memory after the student optimizer step."""

        self.memae_student.memory.update(queries)

    def anomaly_outputs(self, x: Tensor) -> dict[str, Tensor]:
        """Return equations (23)--(27): three maps, combined map, and score."""

        with torch.no_grad():
            teacher_target = self.normalized_teacher(x)
        ae_output = self.ae_student(x)
        memae = self.memae_student(x)
        memae_output = memae["output"]

        teacher_ae_map = (teacher_target - ae_output).square().mean(dim=1)
        teacher_memae_map = (teacher_target - memae_output).square().mean(dim=1)
        student_student_map = (ae_output - memae_output).square().mean(dim=1)
        w1, w2, w3 = self.discrepancy_weights
        anomaly_map = (
            w1 * teacher_ae_map
            + w2 * teacher_memae_map
            + w3 * student_student_map
        )
        score = anomaly_map.mean(dim=(1, 2))
        return {
            "teacher_ae_map": teacher_ae_map,
            "teacher_memae_map": teacher_memae_map,
            "student_student_map": student_student_map,
            "anomaly_map": anomaly_map,
            "score": score,
        }

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.anomaly_outputs(x)


__all__ = [
    "UDMA",
    "UDMAAEStudent",
    "UDMAMemAEStudent",
    "UDMAMemory",
    "UDMATeacher",
]
