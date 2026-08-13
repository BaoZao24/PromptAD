"""Shared OFDMA few-shot protocol support for baseline evaluators."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image

from datasets.ofdma_spectrum import (
    DEFAULT_OFDMA_ROOT,
    JAMMER_TYPES,
    NUM_SUS,
    OFDMASpectrogramPreprocessor,
    build_frame_records,
    build_official_scene_split,
    load_ofdma_labels,
    select_support_scene_ids,
)


def add_ofdma_args(parser) -> None:
    parser.add_argument("--ofdma-root", default=str(DEFAULT_OFDMA_ROOT))
    parser.add_argument("--ofdma-shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--ofdma-preprocess-size", type=int, default=240)
    parser.add_argument(
        "--ofdma-geometry",
        choices=["letterbox", "square_warp"],
        default="letterbox",
    )
    parser.add_argument("--max-test-normal-scenes", type=int, default=0)
    parser.add_argument("--max-test-scenes-per-jammer", type=int, default=0)
    parser.add_argument("--max-sus-per-scene", type=int, default=NUM_SUS)


def _capped(values, limit: int):
    return tuple(values if limit <= 0 else values[:limit])


def _sample(record, args) -> dict:
    return {
        "path": record.image_path,
        "label": int(record.label),
        "category": "ofdma",
        "name": f"ofdma-scene{record.scene_id:05d}-su{record.su_id:02d}",
        "dataset": "ofdma",
        "scene_id": int(record.scene_id),
        "su_id": int(record.su_id),
        "jammer_type": str(record.jammer_type),
        "ofdma_root": str(Path(args.ofdma_root).resolve()),
        "ofdma_preprocess_size": int(args.ofdma_preprocess_size),
        "ofdma_geometry": str(args.ofdma_geometry),
    }


def ofdma_jobs(args) -> list[dict]:
    """Build nested scene-level 1/2/4-shot jobs on the official split."""

    if not 1 <= int(args.max_sus_per_scene) <= NUM_SUS:
        raise ValueError(f"--max-sus-per-scene must be in [1, {NUM_SUS}]")
    shots = [int(value) for value in args.ofdma_shots]
    if not shots or any(value < 1 for value in shots):
        raise ValueError("--ofdma-shots must contain positive integers")

    dataset_root = Path(args.ofdma_root)
    labels = load_ofdma_labels(dataset_root)
    split = build_official_scene_split(labels)
    su_ids = tuple(range(int(args.max_sus_per_scene)))

    normal_test_ids = _capped(split.test_normal, int(args.max_test_normal_scenes))
    test_records = build_frame_records(dataset_root, labels, normal_test_ids, su_ids)
    for jammer_type in JAMMER_TYPES:
        scene_ids = _capped(
            split.test_abnormal_by_type[jammer_type],
            int(args.max_test_scenes_per_jammer),
        )
        test_records.extend(build_frame_records(dataset_root, labels, scene_ids, su_ids))
    test_samples = [_sample(record, args) for record in test_records]

    jobs = []
    for shot in shots:
        support_scene_ids = select_support_scene_ids(split.train_normal, shot, int(args.seed))
        support_records = build_frame_records(
            dataset_root,
            labels,
            support_scene_ids,
            su_ids,
        )
        jobs.append(
            {
                "dataset": "ofdma",
                "category": "ofdma",
                "scene": "official_split",
                "jsr": f"{shot}shot",
                "shot": shot,
                "support_scene_ids": support_scene_ids,
                "train_samples": [_sample(record, args) for record in support_records],
                "test_samples": test_samples,
            }
        )
    return jobs


@lru_cache(maxsize=16)
def _preprocessor(
    dataset_root: str,
    output_size: int,
    geometry: str,
    min_db: float | None = None,
    max_db: float | None = None,
):
    return OFDMASpectrogramPreprocessor(
        dataset_root=dataset_root,
        output_size=output_size,
        geometry=geometry,
        min_db=min_db,
        max_db=max_db,
    )


def load_sample_image(sample: dict) -> Image.Image:
    """Load a sample as RGB, applying official OFDMA physics when required."""

    if sample.get("dataset") != "ofdma":
        return Image.open(sample["path"]).convert("RGB")

    image = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(sample["path"])
    preprocessor = _preprocessor(
        str(sample["ofdma_root"]),
        int(sample["ofdma_preprocess_size"]),
        str(sample["ofdma_geometry"]),
        sample.get("ofdma_min_db"),
        sample.get("ofdma_max_db"),
    )
    processed = preprocessor(image)
    return Image.fromarray(processed, mode="L").convert("RGB")
