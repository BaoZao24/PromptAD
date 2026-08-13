"""Target-scene OFDMA cold-start dataset helpers.

One observation contains all 21 sensing-unit spectrograms.  Support images
come only from normal observations of the same target scene.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from datasets.ofdma_spectrum import (
    JAMMER_TYPES,
    NO_JAMMER,
    NUM_SUS,
    OFDMASpectrogramPreprocessor,
)


DEFAULT_TARGET_SCENE_ROOT = Path(
    "/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v1"
)
VALID_ROLES = {"normal_support", "normal_test", "anomaly_test"}


@dataclass(frozen=True)
class TargetSceneFrameRecord:
    target_scene_id: str
    split: str
    observation_id: str
    role: str
    support_rank: int | None
    su_id: int
    image_path: Path
    label: int
    jammer_type: str


def load_target_scene_manifest(
    dataset_root: str | Path = DEFAULT_TARGET_SCENE_ROOT,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing target-scene manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "target_scene_id",
        "split",
        "observation_id",
        "role",
        "support_rank",
        "su_id",
        "image_path",
        "label",
        "jammer_type",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Target-scene manifest is missing columns: {missing}")
    invalid_roles = sorted(set(manifest["role"].astype(str)) - VALID_ROLES)
    if invalid_roles:
        raise ValueError(f"Target-scene manifest has invalid roles: {invalid_roles}")
    return manifest


def available_target_scenes(manifest: pd.DataFrame, split: str) -> list[str]:
    rows = manifest.loc[manifest["split"].astype(str) == str(split)]
    return sorted(rows["target_scene_id"].astype(str).unique().tolist())


def _limit_test_observations(
    rows: pd.DataFrame,
    max_normal_observations: int,
    max_anomaly_observations_per_type: int,
) -> pd.DataFrame:
    selected_ids: list[str] = []
    normal_ids = sorted(
        rows.loc[rows["role"] == "normal_test", "observation_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    if max_normal_observations > 0:
        normal_ids = normal_ids[:max_normal_observations]
    selected_ids.extend(normal_ids)
    for jammer_type in JAMMER_TYPES:
        anomaly_ids = sorted(
            rows.loc[
                (rows["role"] == "anomaly_test")
                & (rows["jammer_type"].astype(str) == jammer_type),
                "observation_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        if max_anomaly_observations_per_type > 0:
            anomaly_ids = anomaly_ids[:max_anomaly_observations_per_type]
        selected_ids.extend(anomaly_ids)
    return rows.loc[rows["observation_id"].astype(str).isin(selected_ids)]


def build_target_scene_records(
    dataset_root: str | Path,
    manifest: pd.DataFrame,
    target_scene_id: str,
    *,
    split: str,
    role: str,
    shot: int | None = None,
    support_seed: int | None = None,
    max_normal_observations: int = 0,
    max_anomaly_observations_per_type: int = 0,
) -> list[TargetSceneFrameRecord]:
    if role not in {"support", "test"}:
        raise ValueError(f"role must be support or test, got {role}")
    dataset_root = Path(dataset_root)
    rows = manifest.loc[
        (manifest["target_scene_id"].astype(str) == str(target_scene_id))
        & (manifest["split"].astype(str) == str(split))
    ].copy()
    if rows.empty:
        raise ValueError(f"No rows for {split}/{target_scene_id}")

    if role == "support":
        if shot is None or shot < 1:
            raise ValueError("A positive shot is required for support records")
        rows = rows.loc[rows["role"] == "normal_support"]
        if support_seed is None:
            rows = rows.loc[
                pd.to_numeric(rows["support_rank"], errors="coerce")
                <= int(shot)
            ]
        else:
            # Select complete observations (all 21 SUs) using a stable
            # hash-based order.  The same support seed therefore gives the
            # identical scene-level support to both branches, independent of
            # pandas row order or process-level RNG state.
            observation_ids = sorted(rows["observation_id"].astype(str).unique())
            ordered = sorted(
                observation_ids,
                key=lambda observation_id: hashlib.sha256(
                    f"{int(support_seed)}:{target_scene_id}:{observation_id}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            selected = set(ordered[: int(shot)])
            rows = rows.loc[rows["observation_id"].astype(str).isin(selected)]
    else:
        rows = rows.loc[rows["role"].isin(["normal_test", "anomaly_test"])]
        rows = _limit_test_observations(
            rows,
            max_normal_observations=max_normal_observations,
            max_anomaly_observations_per_type=max_anomaly_observations_per_type,
        )

    rows = rows.sort_values(["observation_id", "su_id"])
    records = [
        TargetSceneFrameRecord(
            target_scene_id=str(row.target_scene_id),
            split=str(row.split),
            observation_id=str(row.observation_id),
            role=str(row.role),
            support_rank=(
                None if pd.isna(row.support_rank) else int(row.support_rank)
            ),
            su_id=int(row.su_id),
            image_path=dataset_root / str(row.image_path),
            label=int(row.label),
            jammer_type=str(row.jammer_type),
        )
        for row in rows.itertuples(index=False)
    ]
    validate_target_scene_records(records, role=role, shot=shot)
    return records


def validate_target_scene_records(
    records: list[TargetSceneFrameRecord],
    *,
    role: str,
    shot: int | None,
) -> None:
    if not records:
        raise ValueError(f"No {role} records selected")
    missing = [str(record.image_path) for record in records if not record.image_path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} images; first paths:\n{preview}")

    groups: dict[str, list[TargetSceneFrameRecord]] = {}
    for record in records:
        groups.setdefault(record.observation_id, []).append(record)
    bad_su_counts = {
        observation_id: len(group)
        for observation_id, group in groups.items()
        if len(group) != NUM_SUS or sorted(item.su_id for item in group) != list(range(NUM_SUS))
    }
    if bad_su_counts:
        raise ValueError(f"Each observation must contain {NUM_SUS} unique SUs: {bad_su_counts}")
    for group in groups.values():
        if len({item.label for item in group}) != 1 or len(
            {item.jammer_type for item in group}
        ) != 1:
            raise ValueError(f"Inconsistent labels within observation {group[0].observation_id}")

    if role == "support":
        expected = int(shot) if shot is not None else 0
        if len(groups) != expected:
            raise ValueError(
                f"Expected {expected} support observations, found {len(groups)}"
            )
        if any(record.label != 0 or record.jammer_type != NO_JAMMER for record in records):
            raise ValueError("Support must contain normal observations only")


class TargetSceneOFDMAPathDataset(Dataset):
    def __init__(
        self,
        records: list[TargetSceneFrameRecord],
        preprocessor: OFDMASpectrogramPreprocessor,
    ):
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
        name = (
            f"{record.target_scene_id}::{record.observation_id}::su{record.su_id:02d}"
        )
        return image_bgr, mask, record.label, name, record.jammer_type


def parse_target_scene_name(name: str) -> tuple[str, str, int]:
    try:
        target_scene_id, observation_id, su_part = str(name).split("::")
        return target_scene_id, observation_id, int(su_part.removeprefix("su"))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid target-scene sample name: {name}") from exc
