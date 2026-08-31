#!/usr/bin/env python
"""Adapt the ResAD-2024 code snapshot to image-form RF spectrograms.

This evaluator keeps the important ResAD protocol distinction explicit:

* auxiliary training uses *other* RF signal types;
* the target signal type is unseen during auxiliary training;
* only normal target support images are used to construct the target gallery;
* target normal test images and jammer/anomaly images are evaluated afterwards.

The source implementation is kept under ``references/ResAD``.  This file is
an experiment adapter, not a modification of the official implementation.
For data without pixel masks, an abnormal image is represented by an all-one
image mask for the training/evaluation bookkeeping; image-level metrics are
the primary reported metrics in that case.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pyarrow.ipc as pa_ipc
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
RESAD_ROOT = REPO_ROOT / "references" / "ResAD"
# FrEIA is installed locally so the shared environment does not need to be
# changed just to run this external baseline.
sys.path.insert(0, str(RESAD_ROOT / "_deps"))
sys.path.insert(0, str(RESAD_ROOT))

import timm  # noqa: E402

# The checked-out 2024 code imports the timm 0.6 module paths.  The shared
# environment has timm 1.x, whose implementation moved these modules under
# ``timm.layers``.  Alias the moved modules here instead of modifying the
# downloaded reference source.
import importlib  # noqa: E402
import types  # noqa: E402

for _old, _new in (
    ("timm.models.layers.create_act", "timm.layers.create_act"),
    ("timm.models.layers.helpers", "timm.layers.helpers"),
    ("timm.models.layers.create_attn", "timm.layers.create_attn"),
):
    sys.modules.setdefault(_old, importlib.import_module(_new))

# ResAD's utility module imports the MVTec/VisA dataset classes only for its
# optional path-based reference helper.  Importing those dataset modules would
# pull in imgaug and the complete visual-dataset stack, which is irrelevant to
# this RF adapter.  Provide the minimal class attributes needed by that unused
# helper so the official feature utilities can be reused unchanged.
if "datasets" not in sys.modules:
    _datasets_stub = types.ModuleType("datasets")
    _datasets_stub.__path__ = [str(RESAD_ROOT / "datasets")]
    sys.modules["datasets"] = _datasets_stub
for _module_name, _class_name in (("datasets.mvtec", "MVTEC"), ("datasets.visa", "VISA")):
    if _module_name not in sys.modules:
        _module = types.ModuleType(_module_name)
        _stub_class = type(_class_name, (), {"CLASS_NAMES": ()})
        setattr(_module, _class_name, _stub_class)
        sys.modules[_module_name] = _module

from losses.loss import calculate_log_barrier_bi_occ_loss  # noqa: E402
from models.fc_flow import load_flow_model  # noqa: E402
from models.modules import MultiScaleConv, get_position_encoding  # noqa: E402
from models.vq import MultiScaleVQ  # noqa: E402
from train import train as resad_flow_train  # noqa: E402
from utils import (  # noqa: E402
    BoundaryAverager,
    applying_EFDM,
    get_mc_matched_ref_features,
    get_matched_ref_features,
    get_residual_features,
)
from models.utils import get_logp  # noqa: E402
from losses.utils import get_logp_a  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RF_CATEGORIES = ("burst", "deceptive", "dsss", "pulse", "wideband_pulse")


@dataclass(frozen=True)
class SpectralRecord:
    path: Path
    label: int
    group: str
    mask_path: Path | None = None
    name: str = ""
    image_bgr: np.ndarray | None = None


class SpectralPathDataset(Dataset):
    """Return the tensor/mask tuple expected by the ResAD training code."""

    def __init__(self, records: Sequence[SpectralRecord], image_size: int = 224):
        self.records = list(records)
        self.image_size = int(image_size)
        self.image_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _read_bgr(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def __getitem__(self, index: int):
        record = self.records[index]
        bgr = record.image_bgr if record.image_bgr is not None else self._read_bgr(record.path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.image_transform(Image.fromarray(rgb))

        if record.mask_path is not None and record.mask_path.exists():
            mask = self._read_bgr(record.mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8) * 255
            mask = self.mask_transform(Image.fromarray(mask)).float()
            mask = (mask > 0.5).float()
        elif record.label:
            # This is only a bookkeeping mask for image-only RF datasets.
            mask = torch.ones(1, self.image_size, self.image_size)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return (
            image,
            torch.tensor(record.label, dtype=torch.long),
            mask,
            record.group,
            record.name or record.path.name,
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"))


def _rf_record(
    path: Path,
    *,
    category: str,
    noise_level: str,
    label: int,
    mask_root: Path | None = None,
) -> SpectralRecord:
    mask_path = None
    if label and mask_root is not None:
        mask_name = path.name.replace("_abnormal.png", "_groundtruth.png")
        candidate = mask_root / mask_name
        if candidate.exists():
            mask_path = candidate
    return SpectralRecord(
        path=path,
        label=label,
        group=f"{category}/{noise_level}",
        mask_path=mask_path,
        name=f"{category}/{noise_level}/{path.name}",
    )


def build_rf_protocol(args):
    """Build a cross-signal RF protocol compatible with ResAD's setting."""

    root = Path(args.rf_root)
    target = args.target_category
    noise = args.noise_level
    if target not in RF_CATEGORIES:
        raise ValueError(f"Unknown target category {target!r}")

    normal_root = root / target / "normal" / noise
    abnormal_root = root / target / "abnormal" / noise
    mask_root = root / target / "groundtruth" / noise
    normal = _sorted_pngs(normal_root)
    abnormal = _sorted_pngs(abnormal_root)
    if len(normal) <= args.max_shot:
        raise RuntimeError(f"Not enough target normal images: {normal_root}")
    if not abnormal:
        raise RuntimeError(f"No target abnormal images: {abnormal_root}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(normal)).tolist()
    normal = [normal[i] for i in order]
    max_shot = max(args.shots)
    support_paths = normal[:max_shot]
    test_normal = normal[max_shot:]
    if args.target_normal_limit > 0:
        test_normal = test_normal[: args.target_normal_limit]
    if args.target_abnormal_limit > 0:
        abnormal = abnormal[: args.target_abnormal_limit]

    support = [
        _rf_record(path, category=target, noise_level=noise, label=0)
        for path in support_paths
    ]
    target_test = [
        _rf_record(path, category=target, noise_level=noise, label=0)
        for path in test_normal
    ]
    target_test.extend(
        _rf_record(
            path,
            category=target,
            noise_level=noise,
            label=1,
            mask_root=mask_root,
        )
        for path in abnormal
    )

    # Use other signal types as ResAD's seen/auxiliary classes.  The target
    # category never enters this training pool.
    aux_categories = [x for x in args.aux_categories if x != target]
    aux_records: list[SpectralRecord] = []
    aux_reference_records: dict[str, list[SpectralRecord]] = {}
    for category in aux_categories:
        for level in args.aux_noise_levels:
            nroot = root / category / "normal" / level
            aroot = root / category / "abnormal" / level
            groot = root / category / "groundtruth" / level
            normals = _sorted_pngs(nroot)
            anomalies = _sorted_pngs(aroot)
            if not normals or not anomalies:
                continue
            group = f"{category}/{level}"
            normals = normals[: args.aux_normal_limit]
            anomalies = anomalies[: args.aux_abnormal_limit]
            ref_records = [
                _rf_record(path, category=category, noise_level=level, label=0)
                for path in normals[: args.train_ref_shot]
            ]
            if ref_records:
                aux_reference_records[group] = ref_records
            aux_records.extend(
                _rf_record(path, category=category, noise_level=level, label=0)
                for path in normals
            )
            aux_records.extend(
                _rf_record(
                    path,
                    category=category,
                    noise_level=level,
                    label=1,
                    mask_root=groot,
                )
                for path in anomalies
            )

    if not aux_records:
        raise RuntimeError("Auxiliary RF pool is empty")
    args.support_images_per_shot = 1
    return support, target_test, aux_records, aux_reference_records


def build_rf_public_protocol(args):
    """Use public normal measurement records with RF injected anomalies.

    A public measurement record contains 16 frequency patches.  One
    representative patch is used for the ResAD target reference per selected
    record, so one public ``shot`` still means one normal measurement record
    without expanding the exact all-pairs matcher into an unmanageable gallery.
    """

    root = Path(args.rf_root)
    public_root = root / "RF_Spectrum_Public_Dataset"
    record_dirs = sorted(p for p in public_root.glob("MeasRes_*") if p.is_dir())
    max_shot = max(args.shots)
    if len(record_dirs) <= max_shot:
        raise RuntimeError(f"Not enough public RF records under {public_root}")

    support_dirs = record_dirs[:max_shot]
    test_dirs = record_dirs[max_shot:]
    support_paths = [_sorted_pngs(directory)[0] for directory in support_dirs]
    test_normal_paths = [path for directory in test_dirs for path in _sorted_pngs(directory)]
    if args.target_normal_limit > 0:
        test_normal_paths = test_normal_paths[: args.target_normal_limit]

    target = args.target_category
    noise = args.noise_level
    abnormal_root = root / target / "abnormal" / noise
    mask_root = root / target / "groundtruth" / noise
    abnormal_paths = _sorted_pngs(abnormal_root)
    if args.target_abnormal_limit > 0:
        abnormal_paths = abnormal_paths[: args.target_abnormal_limit]
    if not abnormal_paths:
        raise RuntimeError(f"No public RF target anomalies: {abnormal_root}")

    support = [
        SpectralRecord(
            path=path,
            label=0,
            group="public/normal",
            name=f"public/support/{path.name}",
        )
        for path in support_paths
    ]
    target_test = [
        SpectralRecord(
            path=path,
            label=0,
            group="public/normal",
            name=f"public/test/{path.name}",
        )
        for path in test_normal_paths
    ]
    target_test.extend(
        _rf_record(
            path,
            category=target,
            noise_level=noise,
            label=1,
            mask_root=mask_root,
        )
        for path in abnormal_paths
    )

    aux_records: list[SpectralRecord] = []
    aux_reference_records: dict[str, list[SpectralRecord]] = {}
    aux_normal_paths = [
        path for directory in record_dirs[max_shot : max_shot + 8]
        for path in _sorted_pngs(directory)
    ]
    for category in [x for x in args.aux_categories if x != target]:
        for level in args.aux_noise_levels:
            anomalies = _sorted_pngs(root / category / "abnormal" / level)
            if not anomalies or not aux_normal_paths:
                continue
            group = f"{category}/{level}"
            normal_paths = aux_normal_paths[: args.aux_normal_limit]
            anomaly_paths = anomalies[: args.aux_abnormal_limit]
            refs = [
                SpectralRecord(path=path, label=0, group=group, name=f"aux/{group}/{path.name}")
                for path in normal_paths[: args.train_ref_shot]
            ]
            aux_reference_records[group] = refs
            aux_records.extend(
                SpectralRecord(path=path, label=0, group=group, name=f"aux/{group}/{path.name}")
                for path in normal_paths
            )
            aux_records.extend(
                _rf_record(
                    path,
                    category=category,
                    noise_level=level,
                    label=1,
                    mask_root=root / category / "groundtruth" / level,
                )
                for path in anomaly_paths
            )
    if not aux_records:
        raise RuntimeError("Public RF auxiliary pool is empty")
    args.support_images_per_shot = 1
    return support, target_test, aux_records, aux_reference_records


FEDJAM_LABEL_NAMES = {0: "benign", 1: "pulse", 2: "single_tone", 3: "wideband"}


def _fedjam_rows(split_dir: Path):
    for shard in sorted(split_dir.glob("*.arrow")):
        with shard.open("rb") as handle:
            reader = pa_ipc.open_stream(handle)
            for batch_index, batch in enumerate(reader):
                images = batch.column("image").to_pylist()
                labels = batch.column("label").to_pylist()
                for row_index, (image, label) in enumerate(zip(images, labels)):
                    image_bytes = bytes(image["bytes"])
                    yield image_bytes, int(label), f"{shard.name}:batch{batch_index}:row{row_index}"


def _decode_fedjam_bgr(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _array_record(image_bgr: np.ndarray, label: int, group: str, name: str) -> SpectralRecord:
    return SpectralRecord(
        path=Path("<in-memory>"),
        label=label,
        group=group,
        name=name,
        image_bgr=image_bgr,
    )


def build_fedjam_protocol(args):
    """Build a binary unseen-attack FedJam protocol from Arrow shards."""

    root = Path(args.fedjam_root) / "fedjam_dataset"
    target_label = int(args.fedjam_target_label)
    if target_label not in (1, 2, 3):
        raise ValueError("FedJam target label must be 1 (pulse), 2 (single_tone), or 3 (wideband)")
    max_shot = max(args.shots)
    rng = np.random.default_rng(args.seed)
    support_reservoir: list[tuple[np.ndarray, str]] = []
    benign_seen = 0
    aux_by_label: dict[int, list[SpectralRecord]] = {0: []}
    test_records: list[SpectralRecord] = []
    test_normal_count = 0
    test_abnormal_count = 0

    train_dir = root / "train"
    for image_bytes, label, name in _fedjam_rows(train_dir):
        if label == 0:
            image = _decode_fedjam_bgr(image_bytes)
            benign_seen += 1
            if len(support_reservoir) < max_shot:
                support_reservoir.append((image, name))
            else:
                replacement = int(rng.integers(0, benign_seen))
                if replacement < max_shot:
                    support_reservoir[replacement] = (image, name)
            if len(aux_by_label[0]) < max(args.aux_normal_limit, args.train_ref_shot):
                aux_by_label[0].append(_array_record(image, 0, "benign", name))
        elif label != target_label and len(aux_by_label.setdefault(label, [])) < args.aux_abnormal_limit:
            image = _decode_fedjam_bgr(image_bytes)
            aux_by_label[label].append(
                _array_record(image, 1, FEDJAM_LABEL_NAMES[label], name)
            )

    if len(support_reservoir) < max_shot:
        raise RuntimeError(f"FedJam has only {len(support_reservoir)} benign support rows")

    test_dir = root / "test"
    for image_bytes, label, name in _fedjam_rows(test_dir):
        if label not in (0, target_label):
            continue
        if label == 0:
            if args.target_normal_limit > 0 and test_normal_count >= args.target_normal_limit:
                continue
            test_normal_count += 1
            test_records.append(_array_record(_decode_fedjam_bgr(image_bytes), 0, "fedjam_target", name))
        else:
            if args.target_abnormal_limit > 0 and test_abnormal_count >= args.target_abnormal_limit:
                continue
            test_abnormal_count += 1
            test_records.append(_array_record(_decode_fedjam_bgr(image_bytes), 1, "fedjam_target", name))

    support = [
        _array_record(image, 0, "fedjam_target", name)
        for image, name in support_reservoir
    ]
    aux_records: list[SpectralRecord] = []
    aux_reference_records: dict[str, list[SpectralRecord]] = {}
    benign_refs = aux_by_label[0][: args.train_ref_shot]
    for label, records in aux_by_label.items():
        if label == 0:
            aux_records.extend(records[: args.aux_normal_limit])
            aux_reference_records["benign"] = [
                _array_record(record.image_bgr, 0, "benign", record.name)
                for record in benign_refs
            ]
            continue
        if not records:
            continue
        group = FEDJAM_LABEL_NAMES[label]
        aux_records.extend(records)
        # FedJam has one benign background class rather than per-attack normal
        # images; use the same auxiliary benign reference for each seen attack.
        aux_reference_records[group] = [
            _array_record(record.image_bgr, 0, group, record.name)
            for record in benign_refs
        ]

    if not aux_records or not aux_reference_records:
        raise RuntimeError("FedJam auxiliary pool is empty")
    if not test_records or len({record.label for record in test_records}) < 2:
        raise RuntimeError("FedJam target test subset does not contain both labels")
    args.support_images_per_shot = 1
    args.fedjam_target_name = FEDJAM_LABEL_NAMES[target_label]
    return support, test_records, aux_records, aux_reference_records


def _load_project_ofdma_module():
    module_path = REPO_ROOT / "datasets" / "ofdma_spectrum.py"
    spec = importlib.util.spec_from_file_location("project_ofdma_spectrum", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OFDMA adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_ofdma_protocol(args):
    """Build an unseen-jammer-type OFDMA protocol using official preprocessing."""

    ofdma = _load_project_ofdma_module()
    root = Path(args.ofdma_root)
    labels = pd.read_csv(root / "labels.csv")
    normal_ids = labels.index[labels["jammer_type"] == "no jammer"].tolist()
    target_type = args.ofdma_target_jammer
    if target_type not in ofdma.JAMMER_TYPES:
        raise ValueError(f"Unknown OFDMA jammer type: {target_type}")
    if not 1 <= args.ofdma_support_sus <= ofdma.NUM_SUS:
        raise ValueError(f"ofdma-support-sus must be in [1, {ofdma.NUM_SUS}]")
    target_ids = labels.index[labels["jammer_type"] == target_type].tolist()
    if not target_ids:
        raise RuntimeError(f"No OFDMA scenes for target jammer {target_type}")

    preprocessor = ofdma.OFDMASpectrogramPreprocessor(
        root,
        output_size=args.image_size,
        geometry=args.ofdma_geometry,
    )

    def scene_records(scene_ids, label, group, su_ids=None):
        if su_ids is None:
            su_ids = range(ofdma.NUM_SUS)
        records = []
        for scene_id in scene_ids:
            for su_id in su_ids:
                image_path = root / "Images" / f"spectrogram-{int(scene_id):05d}-{su_id:02d}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(image_path)
                adapted = preprocessor(image)
                bgr = cv2.cvtColor(adapted, cv2.COLOR_GRAY2BGR)
                records.append(
                    _array_record(
                        bgr,
                        label,
                        group,
                        f"ofdma/scene{int(scene_id):05d}/su{su_id:02d}",
                    )
                )
        return records

    rng = np.random.default_rng(args.seed)
    support_scene_ids = rng.permutation(np.asarray(normal_ids[: args.ofdma_train_normal_pool]))[
        : max(args.shots)
    ].tolist()
    # One target support image per scene keeps ResAD's exact all-pairs matcher
    # within memory.  The held-out target test still contains all 21 SUs.
    support = scene_records(
        support_scene_ids,
        0,
        "ofdma_target",
        su_ids=range(args.ofdma_support_sus),
    )

    target_normal_start = args.ofdma_train_normal_pool
    target_normal_ids = normal_ids[
        target_normal_start : target_normal_start + args.ofdma_target_normal_scenes
    ]
    target_records = scene_records(target_normal_ids, 0, "ofdma_target")
    target_anomaly_ids = target_ids[-args.ofdma_target_anomaly_scenes :]
    target_records.extend(scene_records(target_anomaly_ids, 1, "ofdma_target"))

    aux_records = []
    aux_reference_records: dict[str, list[SpectralRecord]] = {}
    aux_normal_ids = normal_ids[
        max(args.shots) : max(args.shots) + args.ofdma_aux_normal_scenes
    ]
    aux_normal = scene_records(aux_normal_ids, 0, "ofdma_normal")
    aux_records.extend(aux_normal)
    normal_refs = aux_normal[: args.train_ref_shot * ofdma.NUM_SUS]
    aux_reference_records["ofdma_normal"] = normal_refs
    for jammer_type in ofdma.JAMMER_TYPES:
        if jammer_type == target_type:
            continue
        jammer_ids = labels.index[labels["jammer_type"] == jammer_type].tolist()
        jammer_ids = jammer_ids[: args.ofdma_aux_anomaly_scenes]
        group = f"ofdma/{jammer_type}"
        aux_records.extend(scene_records(jammer_ids, 1, group))
        aux_reference_records[group] = [
            _array_record(record.image_bgr, 0, group, record.name)
            for record in normal_refs
        ]
    args.support_images_per_shot = args.ofdma_support_sus
    args.ofdma_target_name = target_type
    if not target_records or len({record.label for record in target_records}) < 2:
        raise RuntimeError("OFDMA target subset does not contain both normal and abnormal records")
    return support, target_records, aux_records, aux_reference_records


def _loader(records: Sequence[SpectralRecord], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SpectralPathDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


@torch.inference_mode()
def _encode_records(encoder, records: Sequence[SpectralRecord], device, batch_size: int):
    """Extract and flatten feature maps for each record, grouped by class."""

    result: dict[str, list[list[torch.Tensor]]] = {}
    for images, _labels, _masks, groups, _names in _loader(records, batch_size, False):
        features = encoder(images.to(device, non_blocking=True))
        for i, group in enumerate(groups):
            result.setdefault(group, [[] for _ in features])
            for level, feature in enumerate(features):
                feature_i = feature[i : i + 1]
                _, channels, height, width = feature_i.shape
                flattened = feature_i.permute(0, 2, 3, 1).reshape(-1, channels)
                result[group][level].append(flattened.float().cpu())
    return {
        group: [torch.cat(level_chunks, dim=0).to(device) for level_chunks in levels]
        for group, levels in result.items()
    }


def _make_models(args, device):
    checkpoint = Path(args.pretrained_checkpoint) if args.pretrained_checkpoint else None
    if checkpoint is None:
        default_checkpoints = {
            "wide_resnet50_2": Path(
                "/home/wangbei/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
            ),
            "resnet18": Path(
                "/home/wangbei/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            ),
        }
        checkpoint = default_checkpoints.get(args.backbone)
    use_local_checkpoint = checkpoint is not None and checkpoint.is_file()
    encoder = timm.create_model(
        args.backbone,
        features_only=True,
        out_indices=(1, 2, 3),
        pretrained=not use_local_checkpoint,
    )
    if use_local_checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = encoder.load_state_dict(state, strict=False)
        print(
            f"[backbone] loaded local ImageNet checkpoint {checkpoint} "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    encoder = encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    feat_dims = list(encoder.feature_info.channels())
    vq_ops = MultiScaleVQ(
        num_embeddings=args.num_embeddings,
        channels=feat_dims,
    ).to(device)
    constraintor = MultiScaleConv(feat_dims).to(device)
    flow_args = argparse.Namespace(
        flow_arch="conditional_flow_model",
        coupling_layers=args.coupling_layers,
        clamp_alpha=args.clamp_alpha,
        pos_embed_dim=args.pos_embed_dim,
    )
    estimators = [
        load_flow_model(flow_args, feat_dim).to(device)
        for feat_dim in feat_dims
    ]
    return encoder, vq_ops, constraintor, estimators, feat_dims


def _train_auxiliary(args, encoder, vq_ops, constraintor, estimators, aux_records, aux_refs, device):
    """Train the three official ResAD trainable components on auxiliary RF."""

    loader = _loader(aux_records, args.batch_size, True)
    optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_constraintor = torch.optim.Adam(
        constraintor.parameters(), lr=args.lr, weight_decay=0.0005
    )
    flow_parameters = [p for estimator in estimators for p in estimator.parameters()]
    optimizer_flow = torch.optim.Adam(flow_parameters, lr=args.lr, weight_decay=0.0005)
    boundary_ops = BoundaryAverager(num_levels=3)
    history = []

    for epoch in range(args.epochs):
        vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()
        total_loss = 0.0
        steps = 0
        for images, _labels, masks, groups, _names in loader:
            # VectorQuantizer needs at least one normal feature in a batch.
            if not bool((masks == 0).any()):
                continue
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad():
                features = [x.detach() for x in encoder(images)]
                matched = get_mc_matched_ref_features(features, groups, aux_refs)
                residuals = get_residual_features(features, matched, pos_flag=True)

            level_masks = []
            for residual in residuals:
                _, _, height, width = residual.shape
                level_masks.append(
                    F.interpolate(masks, size=(height, width), mode="nearest").squeeze(1)
                )

            loss_vq = vq_ops(residuals, level_masks, train=True)
            optimizer_vq.zero_grad(set_to_none=True)
            loss_vq.backward()
            optimizer_vq.step()

            residuals_before_neck = [x.detach().clone() for x in residuals]
            neck_features = constraintor(*residuals)
            neck_loss = 0.0
            for level, (neck, original) in enumerate(
                zip(neck_features, residuals_before_neck)
            ):
                _, channels, height, width = neck.shape
                neck_flat = neck.permute(0, 2, 3, 1).reshape(-1, channels)
                original_flat = original.permute(0, 2, 3, 1).reshape(-1, channels)
                mask_flat = level_masks[level].reshape(-1)
                loss_i, _loss_n, _loss_a = calculate_log_barrier_bi_occ_loss(
                    neck_flat, mask_flat, original_flat
                )
                neck_loss = neck_loss + loss_i
            optimizer_constraintor.zero_grad(set_to_none=True)
            neck_loss.backward()
            optimizer_constraintor.step()

            detached_neck = [x.detach().clone() for x in neck_features]
            flow_loss, flow_steps = resad_flow_train(
                argparse.Namespace(
                    feature_levels=3,
                    pos_embed_dim=args.pos_embed_dim,
                    device=device,
                    flow_arch="conditional_flow_model",
                    pos_beta=args.pos_beta,
                    margin_tau=args.margin_tau,
                    bgspp_lambda=args.bgspp_lambda,
                ),
                detached_neck,
                estimators,
                optimizer_flow,
                masks,
                boundary_ops,
                epoch,
                N_batch=args.flow_batch_points,
                FIRST_STAGE_EPOCH=args.first_stage_epochs,
            )
            total_loss += float(loss_vq.item()) + float(neck_loss.item()) + float(flow_loss)
            steps += 1 + int(flow_steps)

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(1, steps),
                "steps": steps,
            }
        )
        print(f"[resad][epoch={epoch}] loss={history[-1]['loss']:.6f} steps={steps}", flush=True)
    return history


@torch.inference_mode()
def _score_records(args, encoder, vq_ops, constraintor, estimators, records, target_refs, device):
    """Run the official ResAD scoring branches and return image scores."""

    vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:
        estimator.eval()
    loader = _loader(records, args.eval_batch_size, False)
    labels: list[int] = []
    logps_normal = [[] for _ in estimators]
    logps_abnormal = [[] for _ in estimators]

    for images, batch_labels, _masks, _groups, _names in loader:
        images = images.to(device, non_blocking=True)
        features = [x.detach() for x in encoder(images)]
        matched = get_matched_ref_features(features, target_refs)
        residuals = get_residual_features(features, matched, pos_flag=True)
        fdm_features = vq_ops(residuals, train=False)
        residuals = applying_EFDM(residuals, fdm_features, alpha=args.fdm_alpha)
        residuals = constraintor(*residuals)
        labels.extend(int(x) for x in batch_labels.tolist())

        for level, (residual, estimator) in enumerate(zip(residuals, estimators)):
            batch_size, channels, height, width = residual.shape
            flattened = residual.permute(0, 2, 3, 1).reshape(-1, channels)
            pos = get_position_encoding(args.pos_embed_dim, height, width).to(device)
            pos = pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            pos = pos.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
            z, logdet = estimator(flattened, [pos])
            logp = get_logp(channels, z, logdet).reshape(batch_size, height, width)
            logp = logp / channels
            logp_a = get_logp_a(channels, z, logdet)
            logits = torch.stack([logp.reshape(-1), logp_a], dim=-1)
            score_a = torch.softmax(logits, dim=-1)[:, 1].reshape(batch_size, height, width)
            logps_normal[level].append(logp.detach().cpu())
            logps_abnormal[level].append(score_a.detach().cpu())

    def convert(logps, abnormal: bool):
        maps = []
        for level_values in logps:
            values = torch.cat(level_values, dim=0)
            if not abnormal:
                values = values - values.max()
                values = torch.exp(values)
            values = F.interpolate(
                values.unsqueeze(1), size=(args.image_size, args.image_size), mode="bilinear", align_corners=True
            ).squeeze(1).numpy()
            maps.append(values)
        combined = np.sum(maps, axis=0)
        if not abnormal:
            combined = combined.max() - combined
        else:
            combined = combined / len(maps)
        return np.stack([gaussian_filter(x, sigma=4) for x in combined], axis=0)

    normal_scores = convert(logps_normal, abnormal=False)
    abnormal_scores = convert(logps_abnormal, abnormal=True)
    merged_scores = (normal_scores + abnormal_scores) / 2.0
    return np.asarray(labels, dtype=np.int64), {
        "logp": np.max(normal_scores.reshape(len(labels), -1), axis=1),
        "bscore": np.max(abnormal_scores.reshape(len(labels), -1), axis=1),
        "merged": np.max(merged_scores.reshape(len(labels), -1), axis=1),
    }


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[valid[0]] * 100.0) if len(valid) else 100.0
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": fpr95,
    }


def _write_outputs(args, rows, train_history, protocol):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "ResAD (NeurIPS 2024 code snapshot)",
        "resad_commit": "08c9b74934c5d25524f6e3bf42761bdb2e6efc34",
        "protocol": protocol,
        "args": vars(args),
        "train_history": train_history,
        "metrics": rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="rf_self",
        choices=("rf_self", "rf_public", "fedjam", "ofdma"),
    )
    parser.add_argument("--rf-root", default="/mnt/data/wangbei/data/RF_SPE_PNG")
    parser.add_argument("--fedjam-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--fedjam-target-label", type=int, default=1)
    parser.add_argument(
        "--ofdma-root",
        default="/mnt/data/wangbei/data/ofdma-spectrum-anomalies-simulation/Example_Use/Dataset",
    )
    parser.add_argument("--ofdma-target-jammer", default="barrage")
    parser.add_argument(
        "--ofdma-geometry", default="letterbox", choices=("letterbox", "square_warp")
    )
    parser.add_argument("--ofdma-train-normal-pool", type=int, default=8800)
    parser.add_argument("--ofdma-target-normal-scenes", type=int, default=2)
    parser.add_argument("--ofdma-target-anomaly-scenes", type=int, default=2)
    parser.add_argument("--ofdma-aux-normal-scenes", type=int, default=4)
    parser.add_argument("--ofdma-aux-anomaly-scenes", type=int, default=1)
    parser.add_argument("--ofdma-support-sus", type=int, default=1)
    parser.add_argument("--target-category", default="burst", choices=RF_CATEGORIES)
    parser.add_argument("--noise-level", default="m10db")
    parser.add_argument("--aux-categories", nargs="+", default=list(RF_CATEGORIES))
    parser.add_argument("--aux-noise-levels", nargs="+", default=["m10db"])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-normal-limit", type=int, default=8)
    parser.add_argument("--aux-abnormal-limit", type=int, default=8)
    parser.add_argument("--target-normal-limit", type=int, default=32)
    parser.add_argument("--target-abnormal-limit", type=int, default=32)
    parser.add_argument("--train-ref-shot", type=int, default=4)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--pretrained-checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.0,
        help="Optional per-process CUDA memory cap; useful on shared GPUs.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--first-stage-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--coupling-layers", type=int, default=1)
    parser.add_argument("--clamp-alpha", type=float, default=1.9)
    parser.add_argument("--pos-embed-dim", type=int, default=32)
    parser.add_argument("--pos-beta", type=float, default=0.05)
    parser.add_argument("--margin-tau", type=float, default=0.1)
    parser.add_argument("--bgspp-lambda", type=float, default=1.0)
    parser.add_argument("--flow-batch-points", type=int, default=512)
    parser.add_argument("--num-embeddings", type=int, default=64)
    parser.add_argument("--fdm-alpha", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/resad/rf_burst_smoke",
    )
    return parser.parse_args()


def main(args):
    if not args.shots or min(args.shots) < 1:
        raise ValueError("shots must be positive")
    args.max_shot = max(args.shots)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and args.max_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device=device)

    if args.dataset == "rf_public":
        support, target_test, aux_records, aux_reference_records = build_rf_public_protocol(args)
    elif args.dataset == "fedjam":
        support, target_test, aux_records, aux_reference_records = build_fedjam_protocol(args)
    elif args.dataset == "ofdma":
        support, target_test, aux_records, aux_reference_records = build_ofdma_protocol(args)
    else:
        support, target_test, aux_records, aux_reference_records = build_rf_protocol(args)
    print(
        f"[protocol] target={args.target_category}/{args.noise_level} "
        f"support_pool={len(support)} target_test={len(target_test)} "
        f"aux_records={len(aux_records)} aux_groups={len(aux_reference_records)}",
        flush=True,
    )
    encoder, vq_ops, constraintor, estimators, feat_dims = _make_models(args, device)
    aux_refs = _encode_records(
        encoder, [record for records in aux_reference_records.values() for record in records], device, args.eval_batch_size
    )
    # _encode_records returns the same nested shape we need, but the key is a
    # group name and it may be flattened across multiple reference images.
    train_history = _train_auxiliary(
        args, encoder, vq_ops, constraintor, estimators, aux_records, aux_refs, device
    )

    support_feature_groups = _encode_records(encoder, support, device, args.eval_batch_size)
    support_group = next(iter(support_feature_groups))
    support_features = support_feature_groups[support_group]
    rows = []
    for shot in args.shots:
        target_refs = [features[:0] for features in support_features]
        # Support features are stored as flattened patch rows.  Each support
        # image contributes the same number of patches at a given level.
        for level, features in enumerate(support_features):
            _, channels, height, width = encoder(torch.zeros(1, 3, args.image_size, args.image_size, device=device))[level].shape
            patches_per_image = height * width
            support_images = shot * args.support_images_per_shot
            target_refs[level] = features[: support_images * patches_per_image]
        labels, score_branches = _score_records(
            args, encoder, vq_ops, constraintor, estimators, target_test, target_refs, device
        )
        for branch, scores in score_branches.items():
            values = _metrics(labels, scores)
            rows.append({"shot": shot, "branch": branch, **values})
            print(
                f"[result] shot={shot} branch={branch} "
                f"AUROC={values['auroc']:.3f} AUPRC={values['auprc']:.3f} "
                f"FPR95={values['fpr95']:.3f}",
                flush=True,
            )

    protocol = {
        "dataset": (
            "RF Spectrum public dataset"
            if args.dataset == "rf_public"
            else "FedJam"
            if args.dataset == "fedjam"
            else "OFDMA spectrum anomalies simulation"
            if args.dataset == "ofdma"
            else "RF_SPE_PNG self RF"
        ),
        "target": (
            getattr(args, "fedjam_target_name", "")
            if args.dataset == "fedjam"
            else getattr(args, "ofdma_target_name", "")
            if args.dataset == "ofdma"
            else f"{args.target_category}/{args.noise_level}"
        ),
        "auxiliary_categories": (
            [x for x in args.aux_categories if x != args.target_category]
            if args.dataset in {"rf_self", "rf_public"}
            else [
                x
                for x in FEDJAM_LABEL_NAMES.values()
                if x != getattr(args, "fedjam_target_name", "")
            ]
            if args.dataset == "fedjam"
            else [
                x
                for x in ("barrage", "deceptive", "pilot", "sweep", "random_hop")
                if x != getattr(args, "ofdma_target_name", "")
            ]
        ),
        "auxiliary_training": (
            "other RF signal categories only"
            if args.dataset in {"rf_self", "rf_public"}
            else "other FedJam attack labels and benign"
            if args.dataset == "fedjam"
            else "other OFDMA jammer types and no-jammer scenes"
        ),
        "target_support": "normal images only; nested 1/2/4-shot subsets",
        "target_test": "remaining target normal images plus target abnormal images",
        "backbone": args.backbone,
        "feature_dims": feat_dims,
        "image_metrics": "max pooled merged ResAD score; image-level AUROC/AUPRC/FPR@95%TPR",
        "pixel_mask_note": "RF groundtruth masks used when available",
    }
    _write_outputs(args, rows, train_history, protocol)


if __name__ == "__main__":
    main(parse_args())
