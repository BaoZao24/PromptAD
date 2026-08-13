"""OFDMA spectrum dataset support for normal-only few-shot CLS evaluation.

The PNG layout is ``height=1320 frequency bins`` by ``width=70 time bins``.
The official example code aggregates every 12 subcarriers in the linear-power
domain before normalizing the result.  This module keeps that physical
preprocessing while streaming images from disk instead of caching the 45 GB
dataset in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


DEFAULT_OFDMA_ROOT = Path(
    "/mnt/data/wangbei/data/ofdma-spectrum-anomalies-simulation/Example_Use/Dataset"
)
JAMMER_TYPES = ("barrage", "deceptive", "pilot", "sweep", "random_hop")
NO_JAMMER = "no jammer"
NUM_SUS = 21


@dataclass(frozen=True)
class OFDMASceneSplit:
    train_normal: tuple[int, ...]
    valid_normal: tuple[int, ...]
    test_normal: tuple[int, ...]
    test_abnormal_by_type: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class OFDMAFrameRecord:
    scene_id: int
    su_id: int
    image_path: Path
    label: int
    jammer_type: str


def load_ofdma_labels(dataset_root: str | Path = DEFAULT_OFDMA_ROOT) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    labels_path = dataset_root / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing OFDMA labels: {labels_path}")
    labels = pd.read_csv(labels_path)
    if "jammer_type" not in labels:
        raise ValueError(f"{labels_path} does not contain jammer_type")
    labels = labels.copy()
    labels.insert(0, "scene_id", np.arange(len(labels), dtype=np.int32))
    return labels


def build_official_scene_split(
    labels: pd.DataFrame,
    num_train_normal: int = 8800,
    num_valid_normal: int = 200,
    num_test_normal: int = 1000,
    num_test_anomaly: int = 2500,
) -> OFDMASceneSplit:
    """Reproduce the official unsupervised split from Evaluation.ipynb."""

    normal_ids = labels.loc[labels["jammer_type"] == NO_JAMMER, "scene_id"].astype(int).tolist()
    if len(normal_ids) < num_train_normal + num_valid_normal + num_test_normal:
        raise ValueError(
            "Not enough normal scenes for the official split: "
            f"available={len(normal_ids)} requested="
            f"{num_train_normal + num_valid_normal + num_test_normal}"
        )

    per_jammer = num_test_anomaly // len(JAMMER_TYPES)
    if per_jammer * len(JAMMER_TYPES) != num_test_anomaly:
        raise ValueError("num_test_anomaly must be divisible by the number of jammer types")

    abnormal_by_type: dict[str, tuple[int, ...]] = {}
    for jammer_type in JAMMER_TYPES:
        scene_ids = labels.loc[labels["jammer_type"] == jammer_type, "scene_id"].astype(int).tolist()
        if len(scene_ids) < per_jammer:
            raise ValueError(
                f"Not enough {jammer_type} scenes: available={len(scene_ids)} requested={per_jammer}"
            )
        abnormal_by_type[jammer_type] = tuple(scene_ids[-per_jammer:])

    split = OFDMASceneSplit(
        train_normal=tuple(normal_ids[:num_train_normal]),
        valid_normal=tuple(normal_ids[num_train_normal:num_train_normal + num_valid_normal]),
        test_normal=tuple(normal_ids[-num_test_normal:]),
        test_abnormal_by_type=abnormal_by_type,
    )
    assert not (set(split.train_normal) & set(split.test_normal))
    assert not (set(split.valid_normal) & set(split.test_normal))
    return split


def select_support_scene_ids(
    train_normal_scene_ids: tuple[int, ...] | list[int],
    shots: int,
    seed: int,
) -> tuple[int, ...]:
    """Select nested, deterministic scene-level shots from the official train pool."""

    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    scene_ids = np.asarray(train_normal_scene_ids, dtype=np.int32)
    if shots > len(scene_ids):
        raise ValueError(f"shots={shots} exceeds train normal scenes={len(scene_ids)}")
    order = np.random.default_rng(seed).permutation(len(scene_ids))
    return tuple(int(x) for x in scene_ids[order[:shots]])


def build_frame_records(
    dataset_root: str | Path,
    labels: pd.DataFrame,
    scene_ids: list[int] | tuple[int, ...],
    su_ids: list[int] | tuple[int, ...] = tuple(range(NUM_SUS)),
) -> list[OFDMAFrameRecord]:
    dataset_root = Path(dataset_root)
    images_root = dataset_root / "Images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing OFDMA image directory: {images_root}")

    label_by_scene = labels.set_index("scene_id")["jammer_type"].to_dict()
    records: list[OFDMAFrameRecord] = []
    for scene_id in scene_ids:
        jammer_type = str(label_by_scene[int(scene_id)])
        label = int(jammer_type != NO_JAMMER)
        for su_id in su_ids:
            if not 0 <= int(su_id) < NUM_SUS:
                raise ValueError(f"Invalid SU id: {su_id}")
            image_path = images_root / f"spectrogram-{int(scene_id):05d}-{int(su_id):02d}.png"
            records.append(
                OFDMAFrameRecord(
                    scene_id=int(scene_id),
                    su_id=int(su_id),
                    image_path=image_path,
                    label=label,
                    jammer_type=jammer_type,
                )
            )
    return records


class OFDMASpectrogramPreprocessor:
    """Official subcarrier aggregation followed by square model adaptation."""

    def __init__(
        self,
        dataset_root: str | Path = DEFAULT_OFDMA_ROOT,
        output_size: int = 240,
        subcarriers_per_rb: int = 12,
        geometry: str = "letterbox",
        min_db: float | None = None,
        max_db: float | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.output_size = int(output_size)
        self.subcarriers_per_rb = int(subcarriers_per_rb)
        self.geometry = geometry
        if self.output_size < 1:
            raise ValueError("output_size must be positive")
        if self.subcarriers_per_rb < 1:
            raise ValueError("subcarriers_per_rb must be positive")
        if self.geometry not in {"letterbox", "square_warp"}:
            raise ValueError(f"Unsupported OFDMA geometry: {self.geometry}")

        if (min_db is None) != (max_db is None):
            raise ValueError("min_db and max_db must be provided together")
        if min_db is None:
            minmax_path = self.dataset_root / "spectrogram_min_max.csv"
            if not minmax_path.is_file():
                raise FileNotFoundError(f"Missing OFDMA spectrogram range: {minmax_path}")
            minmax = pd.read_csv(minmax_path)
            self.min_db = float(minmax["min_val"].iloc[0])
            self.max_db = float(minmax["max_val"].iloc[0])
        else:
            self.min_db = float(min_db)
            self.max_db = float(max_db)
        if not self.max_db > self.min_db:
            raise ValueError(
                f"Invalid OFDMA spectrogram range: min_db={self.min_db}, max_db={self.max_db}"
            )
        power_gain_db = 10.0 * np.log10(float(self.subcarriers_per_rb))
        self.aggregated_min_db = self.min_db + power_gain_db
        self.aggregated_max_db = self.max_db + power_gain_db

    def aggregate_subcarriers(self, image_u8: np.ndarray) -> np.ndarray:
        if image_u8.ndim != 2:
            raise ValueError(f"Expected grayscale OFDMA image, got shape={image_u8.shape}")
        freq_bins, time_bins = image_u8.shape
        if time_bins != 70:
            raise ValueError(
                "OFDMA axis mismatch: expected width=70 time bins and height=frequency bins, "
                f"got shape={image_u8.shape}"
            )
        if freq_bins % self.subcarriers_per_rb != 0:
            raise ValueError(
                f"Frequency bins {freq_bins} are not divisible by {self.subcarriers_per_rb}"
            )

        spectrogram_db = (
            image_u8.astype(np.float64) / 255.0 * (self.max_db - self.min_db) + self.min_db
        )
        grouped = spectrogram_db.reshape(
            freq_bins // self.subcarriers_per_rb,
            self.subcarriers_per_rb,
            time_bins,
        )
        grouped_power = np.power(10.0, grouped / 10.0).sum(axis=1)
        aggregated_db = 10.0 * np.log10(np.maximum(grouped_power, np.finfo(np.float64).tiny))
        normalized = (aggregated_db - self.aggregated_min_db) / (
            self.aggregated_max_db - self.aggregated_min_db
        )
        return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)

    def adapt_geometry(self, aggregated_u8: np.ndarray) -> np.ndarray:
        size = self.output_size
        if self.geometry == "square_warp":
            return cv2.resize(aggregated_u8, (size, size), interpolation=cv2.INTER_LINEAR)

        height, width = aggregated_u8.shape
        scale = min(size / float(width), size / float(height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            aggregated_u8,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_left = (size - resized_width) // 2
        pad_right = size - resized_width - pad_left
        pad_top = (size - resized_height) // 2
        pad_bottom = size - resized_height - pad_top
        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    def __call__(self, image_u8: np.ndarray) -> np.ndarray:
        return self.adapt_geometry(self.aggregate_subcarriers(image_u8))


class OFDMAPathDataset(Dataset):
    def __init__(self, records: list[OFDMAFrameRecord], preprocessor: OFDMASpectrogramPreprocessor):
        self.records = records
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(record.image_path)
        image = self.preprocessor(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(image.shape, dtype=np.uint8)
        name = f"ofdma-scene{record.scene_id:05d}-su{record.su_id:02d}"
        return image_bgr, mask, record.label, name, record.jammer_type


def parse_ofdma_name(name: str) -> tuple[int, int]:
    try:
        scene_part, su_part = str(name).rsplit("-", 2)[-2:]
        return int(scene_part.removeprefix("scene")), int(su_part.removeprefix("su"))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid OFDMA sample name: {name}") from exc
