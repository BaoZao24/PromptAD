"""Small, self-contained GRETEL-style graph detector for spectrogram images.

This module implements the *spectrogram adaptation* used by
``eval_gretel_spectral.py``.  It intentionally does not pretend that a PNG
spectrogram is raw IQ: the paper's physical SNR feature is replaced by a
support-independent image-domain band-contrast feature.  The graph and the
teacher/student/memory design otherwise follow the paper at a functional
level:

* split a 64x256 time-frequency map into 16 contiguous frequency nodes;
* connect each node to itself and its immediate frequency neighbours;
* train a teacher graph autoencoder on normal support only;
* freeze the teacher and train a smaller memory-augmented student; and
* score reconstruction, embedding, and attention disagreement.

No test image is ever used by ``fit_gretel``.  The implementation uses dense
16-node attention deliberately: it is dependency-free and considerably
lighter than introducing a graph framework for this fixed topology.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GretelConfig:
    """Frozen adaptation settings, chosen before any test-set evaluation."""

    time_bins: int = 64
    frequency_bins: int = 256
    frequency_nodes: int = 16
    hidden_dim: int = 64
    teacher_heads: int = 4
    student_heads: int = 2
    memory_items: int = 16
    teacher_epochs: int = 100
    student_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    reconstruction_weight: float = 1.0
    embedding_weight: float = 1.0
    attention_weight: float = 1.0
    memory_weight: float = 0.1

    @property
    def patch_width(self) -> int:
        if self.frequency_bins % self.frequency_nodes:
            raise ValueError("frequency_bins must be divisible by frequency_nodes")
        return self.frequency_bins // self.frequency_nodes

    def serializable(self) -> dict:
        return asdict(self)


def image_to_power_map(image: np.ndarray, config: GretelConfig) -> np.ndarray:
    """Convert an RGB/BGR/grayscale spectrogram image to a fixed power map.

    Pixel intensity is retained on [0, 1] rather than independently
    standardising each image, because per-image standardisation would erase a
    broad-band power change that can itself be anomalous.  For colour
    spectrogram PNGs, luminance is the available image-domain power proxy.
    """

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[..., :3]
        if array.shape[2] != 3:
            raise ValueError(f"Expected 3 colour channels, got {array.shape}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram image, got {array.shape}")
    resized = cv2.resize(
        array,
        (config.frequency_bins, config.time_bins),
        interpolation=cv2.INTER_AREA,
    )
    if resized.dtype == np.uint8:
        return np.ascontiguousarray(resized.astype(np.float32) / 255.0)
    resized = resized.astype(np.float32)
    if resized.size and (resized.min() < 0.0 or resized.max() > 1.0):
        # OFDMA preprocessing can return a physical dB-normalised uint-like
        # image.  This is a fixed range conversion, not test-set fitting.
        resized = np.clip(resized, 0.0, 255.0) / 255.0
    return np.ascontiguousarray(resized)


def power_maps_to_graph_inputs(
    maps: torch.Tensor, config: GretelConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return paper-inspired node statistics and the reconstructable patches."""

    if maps.ndim != 3:
        raise ValueError(f"Expected [batch,time,frequency], got {tuple(maps.shape)}")
    batch, height, width = maps.shape
    if (height, width) != (config.time_bins, config.frequency_bins):
        raise ValueError(
            f"Expected {(config.time_bins, config.frequency_bins)}, got {(height, width)}"
        )
    patches = maps.reshape(
        batch,
        config.time_bins,
        config.frequency_nodes,
        config.patch_width,
    ).permute(0, 2, 1, 3).contiguous()
    mean = patches.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
    std = patches.std(dim=(2, 3), unbiased=False, keepdim=False).unsqueeze(-1)
    # A relative energy contrast is the only defensible substitute for the
    # paper's segment SNR when raw IQ/noise-floor calibration is unavailable.
    global_median = maps.flatten(1).median(dim=1).values[:, None, None]
    global_iqr = (
        torch.quantile(maps.flatten(1), 0.75, dim=1)
        - torch.quantile(maps.flatten(1), 0.25, dim=1)
    )[:, None, None].clamp_min(1e-3)
    contrast = (mean - global_median) / global_iqr
    positions = torch.arange(
        config.frequency_nodes, device=maps.device, dtype=maps.dtype
    )
    positions = positions / max(config.frequency_nodes - 1, 1)
    position_encoding = torch.stack(
        (
            torch.sin(np.pi * positions),
            torch.cos(np.pi * positions),
            torch.sin(2.0 * np.pi * positions),
            torch.cos(2.0 * np.pi * positions),
        ),
        dim=-1,
    ).unsqueeze(0).expand(batch, -1, -1)
    features = torch.cat((mean, std, contrast, position_encoding), dim=-1)
    return features, patches.flatten(2)


def frequency_adjacency(nodes: int, device: torch.device) -> torch.Tensor:
    """Undirected self + immediate-neighbour frequency-chain adjacency."""

    adjacency = torch.eye(nodes, device=device, dtype=torch.bool)
    if nodes > 1:
        indices = torch.arange(nodes - 1, device=device)
        adjacency[indices, indices + 1] = True
        adjacency[indices + 1, indices] = True
    return adjacency


class DenseGraphAttention(nn.Module):
    """Multi-head GAT for the fixed small frequency graph."""

    def __init__(self, in_dim: int, out_dim: int, heads: int):
        super().__init__()
        if out_dim % heads:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.output = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, values: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, _ = values.shape
        query = self.query(values).reshape(batch, nodes, self.heads, self.head_dim)
        key = self.key(values).reshape(batch, nodes, self.heads, self.head_dim)
        value = self.value(values).reshape(batch, nodes, self.heads, self.head_dim)
        logits = torch.einsum("bihd,bjhd->bhij", query, key)
        logits = logits / float(self.head_dim) ** 0.5
        mask = adjacency.unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhij,bjhd->bihd", attention, value)
        attended = attended.reshape(batch, nodes, -1)
        output = self.norm(values if values.shape[-1] == attended.shape[-1] else 0.0)
        output = output + self.output(attended)
        return F.gelu(output), attention.mean(dim=1)


class GraphBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.layers = nn.ModuleList(
            [DenseGraphAttention(hidden_dim, hidden_dim, heads) for _ in range(layers)]
        )
        self.pool = nn.Linear(hidden_dim, 1)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.input(node_features)
        attention = None
        for layer in self.layers:
            values, attention = layer(values, adjacency)
        weights = torch.softmax(self.pool(values).squeeze(-1), dim=1)
        graph_embedding = torch.einsum("bn,bnd->bd", weights, values)
        if attention is None:
            raise RuntimeError("GraphBackbone has no attention layers")
        return values, graph_embedding, attention


class GretelSpectrogramModel(nn.Module):
    """Teacher/student GATs, memory prototypes, and gated cross attention."""

    def __init__(self, config: GretelConfig):
        super().__init__()
        self.config = config
        self.teacher = GraphBackbone(7, config.hidden_dim, config.teacher_heads, layers=3)
        self.student = GraphBackbone(7, config.hidden_dim, config.student_heads, layers=2)
        self.teacher_decoder = nn.Linear(config.hidden_dim, config.time_bins * config.patch_width)
        self.student_decoder = nn.Linear(config.hidden_dim, config.time_bins * config.patch_width)
        self.memory = nn.Parameter(torch.randn(config.memory_items, config.hidden_dim) * 0.02)
        self.cross_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.cross_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.cross_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.student_pool = nn.Linear(config.hidden_dim, 1)

    def teacher_forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        nodes, graph, attention = self.teacher(node_features, adjacency)
        return {
            "nodes": nodes,
            "graph": graph,
            "attention": attention,
            "reconstruction": self.teacher_decoder(nodes),
        }

    def student_forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        teacher_nodes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nodes, _graph, attention = self.student(node_features, adjacency)
        memory = F.normalize(self.memory, dim=-1)
        similarity = torch.einsum("bnd,md->bnm", F.normalize(nodes, dim=-1), memory)
        memory_weights = torch.softmax(similarity, dim=-1)
        memory_read = torch.einsum("bnm,md->bnd", memory_weights, self.memory)

        query = self.cross_query(nodes)
        key = self.cross_key(teacher_nodes)
        value = self.cross_value(teacher_nodes)
        cross_logits = torch.einsum("bnd,bmd->bnm", query, key) / float(nodes.shape[-1]) ** 0.5
        cross_weights = torch.softmax(cross_logits, dim=-1)
        cross_read = torch.einsum("bnm,bmd->bnd", cross_weights, value)
        gate = self.gate(torch.cat((nodes, memory_read, cross_read), dim=-1))
        fused = gate * nodes + (1.0 - gate) * (memory_read + cross_read) * 0.5
        pool_weights = torch.softmax(self.student_pool(fused).squeeze(-1), dim=1)
        graph = torch.einsum("bn,bnd->bd", pool_weights, fused)
        return {
            "nodes": fused,
            "graph": graph,
            "attention": attention,
            "memory_read": memory_read,
            "reconstruction": self.student_decoder(fused),
        }

    def student_parameters(self):
        modules = [
            self.student,
            self.student_decoder,
            self.cross_query,
            self.cross_key,
            self.cross_value,
            self.gate,
            self.student_pool,
        ]
        parameters = [self.memory]
        for module in modules:
            parameters.extend(module.parameters())
        return parameters


def _batches(values: torch.Tensor, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    indices = torch.randperm(values.shape[0], device=values.device) if shuffle else torch.arange(
        values.shape[0], device=values.device
    )
    return [values[indices[start : start + batch_size]] for start in range(0, len(indices), batch_size)]


def _teacher_loss(model, maps: torch.Tensor, adjacency: torch.Tensor, config: GretelConfig) -> torch.Tensor:
    features, patches = power_maps_to_graph_inputs(maps, config)
    output = model.teacher_forward(features, adjacency)
    return F.mse_loss(output["reconstruction"], patches)


def _student_losses(
    model: GretelSpectrogramModel,
    maps: torch.Tensor,
    adjacency: torch.Tensor,
    config: GretelConfig,
) -> dict[str, torch.Tensor]:
    features, patches = power_maps_to_graph_inputs(maps, config)
    with torch.no_grad():
        teacher = model.teacher_forward(features, adjacency)
    student = model.student_forward(features, adjacency, teacher["nodes"])
    reconstruction = F.mse_loss(student["reconstruction"], patches)
    embedding = F.mse_loss(F.normalize(student["graph"], dim=-1), F.normalize(teacher["graph"], dim=-1))
    attention = F.mse_loss(student["attention"], teacher["attention"])
    memory = F.mse_loss(student["nodes"], student["memory_read"])
    total = (
        config.reconstruction_weight * reconstruction
        + config.embedding_weight * embedding
        + config.attention_weight * attention
        + config.memory_weight * memory
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "embedding": embedding,
        "attention": attention,
        "memory": memory,
    }


def fit_gretel(
    support_maps: np.ndarray,
    config: GretelConfig,
    device: torch.device,
    seed: int,
) -> tuple[GretelSpectrogramModel, dict[str, float]]:
    """Fit exactly on normal support maps; no validation/test access occurs."""

    if len(support_maps) < 1:
        raise ValueError("GRETEL needs at least one normal support image")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    support = torch.as_tensor(support_maps, dtype=torch.float32, device=device)
    model = GretelSpectrogramModel(config).to(device)
    adjacency = frequency_adjacency(config.frequency_nodes, device)

    model.train()
    teacher_optimizer = torch.optim.AdamW(
        list(model.teacher.parameters()) + list(model.teacher_decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    teacher_final = float("nan")
    for _epoch in range(config.teacher_epochs):
        losses = []
        for batch in _batches(support, config.batch_size, shuffle=True):
            teacher_optimizer.zero_grad(set_to_none=True)
            loss = _teacher_loss(model, batch, adjacency, config)
            loss.backward()
            teacher_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        teacher_final = float(np.mean(losses))

    for parameter in model.teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.teacher_decoder.parameters():
        parameter.requires_grad_(False)
    student_optimizer = torch.optim.AdamW(
        model.student_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    student_final: dict[str, float] = {}
    for _epoch in range(config.student_epochs):
        epoch_values: dict[str, list[float]] = {key: [] for key in ("total", "reconstruction", "embedding", "attention", "memory")}
        for batch in _batches(support, config.batch_size, shuffle=True):
            student_optimizer.zero_grad(set_to_none=True)
            losses = _student_losses(model, batch, adjacency, config)
            losses["total"].backward()
            student_optimizer.step()
            for key, value in losses.items():
                epoch_values[key].append(float(value.detach().cpu()))
        student_final = {key: float(np.mean(values)) for key, values in epoch_values.items()}

    model.eval()
    return model, {"teacher_reconstruction": teacher_final, **{f"student_{key}": value for key, value in student_final.items()}}


@torch.inference_mode()
def score_gretel_maps(
    model: GretelSpectrogramModel,
    maps: np.ndarray | torch.Tensor,
    config: GretelConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score a batch of maps with the fixed three-term GRETEL score."""

    values = torch.as_tensor(maps, dtype=torch.float32, device=device)
    adjacency = frequency_adjacency(config.frequency_nodes, device)
    features, patches = power_maps_to_graph_inputs(values, config)
    teacher = model.teacher_forward(features, adjacency)
    student = model.student_forward(features, adjacency, teacher["nodes"])
    reconstruction = (student["reconstruction"] - patches).pow(2).mean(dim=(1, 2))
    embedding = (
        F.normalize(teacher["graph"], dim=-1) - F.normalize(student["graph"], dim=-1)
    ).pow(2).mean(dim=1)
    attention = (teacher["attention"] - student["attention"]).pow(2).mean(dim=(1, 2))
    score = reconstruction + embedding + attention
    return {
        "score": score.detach().cpu().numpy().astype(np.float32),
        "reconstruction": reconstruction.detach().cpu().numpy().astype(np.float32),
        "embedding": embedding.detach().cpu().numpy().astype(np.float32),
        "attention": attention.detach().cpu().numpy().astype(np.float32),
    }
