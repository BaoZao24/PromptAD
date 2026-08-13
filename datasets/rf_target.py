"""Canonical data access for the self-collected RF target dataset.

The injected signal type and JSR are experimental conditions; the normal
background belongs to the scene.  Formal few-shot experiments therefore build
one support set per scene and evaluate each signal/JSR cell separately.

Raw spectrogram windows use a length of 4000 and a stride of 2000.  We reserve
the first window [0, 4000) for support and only evaluate windows starting at
4000 or later.  The overlapping [2000, 6000) window is intentionally excluded.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


RF_TARGET_ROOT = Path(
    os.environ.get("PROMPTAD_RF_TARGET_ROOT", "/mnt/data/wangbei/data/datasets")
).expanduser()

RF_TARGET_SIGNALS = (
    "burst_signal",
    "chirp_signal",
    "dsss_signal",
    "pulse_signal",
    "deceptive_signal",
)
RF_SCENES = (
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
)
RF_JSR_BY_SIGNAL = {
    "burst_signal": ("m10db", "m20db", "m30db"),
    "chirp_signal": ("m10db", "m20db", "m30db"),
    "dsss_signal": ("m10db", "m20db", "m30db"),
    "pulse_signal": ("m20db", "m30db", "m40db"),
    "deceptive_signal": ("strong", "medium", "weak"),
}
RF_SIGNAL_DIRS = {
    "burst_signal": "burst",
    "chirp_signal": "chirp",
    "dsss_signal": "dsss",
    "pulse_signal": "pulse",
    "deceptive_signal": "deceptive",
}
RF_DECEPTIVE_STORAGE_JSR = {
    "WeaponMuseum_spectrum": {
        "strong": "m10db",
        "medium": "m20db",
        "weak": "m30db",
    },
    "Playground_spectrum": {
        "strong": "m10db",
        "medium": "m20db",
        "weak": "m30db",
    },
    "TimeSquare_spectrum": {
        "strong": "m10db",
        "medium": "m20db",
        "weak": "m30db",
    },
    "Gymnasium_spectrum": {
        "strong": "0db",
        "medium": "10db",
        "weak": "20db",
    },
}

SUPPORT_TIME_RANGE = (0, 4000)
TEST_MIN_TIME_START = SUPPORT_TIME_RANGE[1]

_TIME_RANGE_RE = re.compile(r"_t(\d+)-(\d+)_")
_FREQUENCY_RE = re.compile(r"_f(\d+\.\d+)-(\d+\.\d+)MHz_")


def extract_time_range(path: str | Path) -> tuple[int, int] | None:
    match = _TIME_RANGE_RE.search(Path(path).name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def extract_frequency_band(path: str | Path) -> str | None:
    match = _FREQUENCY_RE.search(Path(path).name)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def rf_cell_roots(
    signal: str,
    scene: str,
    jsr: str,
    *,
    dataset_root: str | Path = RF_TARGET_ROOT,
) -> tuple[Path, Path, Path]:
    try:
        signal_dir = RF_SIGNAL_DIRS[signal]
    except KeyError as exc:
        raise ValueError(f"Unknown RF target signal: {signal!r}") from exc
    storage_jsr = jsr
    if signal == "deceptive_signal":
        try:
            storage_jsr = RF_DECEPTIVE_STORAGE_JSR[scene][jsr]
        except KeyError as exc:
            raise ValueError(
                f"Unknown deceptive strength for scene {scene!r}: {jsr!r}"
            ) from exc
    cell_root = Path(dataset_root) / signal_dir / scene
    return (
        cell_root / "normal" / storage_jsr,
        cell_root / "abnormal" / storage_jsr,
        cell_root / "groundtruth" / storage_jsr,
    )


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {path}")


def _filter_paths(
    paths: Iterable[Path],
    *,
    frequency_band: str | None = None,
    exact_time_range: tuple[int, int] | None = None,
    min_time_start: int | None = None,
) -> list[Path]:
    selected = []
    for path in sorted(paths):
        if frequency_band is not None and extract_frequency_band(path) != frequency_band:
            continue
        time_range = extract_time_range(path)
        if time_range is None:
            raise ValueError(f"RF filename has no time range: {path}")
        if exact_time_range is not None and time_range != exact_time_range:
            continue
        if min_time_start is not None and time_range[0] < min_time_start:
            continue
        selected.append(path)
    return selected


def collect_rf_target_samples(
    signal: str,
    scene: str,
    jsr: str,
    phase: str,
    k_shot: int = 1,
    exclude_paths: Iterable[str | Path] | None = None,
):
    """Return the five-field sample tuples used by RF evaluation tools.

    ``phase="train"`` exposes all normal candidates. The formal target-scene
    manifest owns support selection and time isolation. ``phase="test"``
    exposes the complete cell; the manifest restricts the final test subset.
    """

    normal_root, abnormal_root, gt_root = rf_cell_roots(signal, scene, jsr)
    _require_directory(normal_root, "RF normal")
    excluded = {str(Path(path)) for path in (exclude_paths or [])}

    if phase == "train":
        image_paths = _filter_paths(normal_root.glob("*.png"))
        if k_shot > 0:
            image_paths = image_paths[:k_shot]
        raw = [
            (str(path), 0, 0, f"{scene}_normal")
            for path in image_paths
        ]
    elif phase == "test":
        _require_directory(abnormal_root, "RF abnormal")
        normal_paths = [
            path
            for path in _filter_paths(normal_root.glob("*.png"))
            if str(path) not in excluded
        ]
        abnormal_paths = _filter_paths(abnormal_root.glob("*.png"))
        raw = [
            (str(path), 0, 0, f"{scene}_normal")
            for path in normal_paths
        ]
        for path in abnormal_paths:
            gt_name = path.name.replace("_abnormal.png", "_groundtruth.png")
            gt_path = gt_root / gt_name
            raw.append(
                (
                    str(path),
                    str(gt_path) if gt_path.exists() else 0,
                    1,
                    f"{scene}_abnormal",
                )
            )
    else:
        raise ValueError(f"Unsupported RF phase: {phase!r}")

    prefix = f"{signal}-{scene}-{jsr}"
    return [
        (image_path, gt, label, sample_type, prefix)
        for image_path, gt, label, sample_type in raw
    ]


def load_legacy_rf_signal(
    signal: str,
    category: str,
    k_shot: int,
    *,
    noise_level: str,
    freq: str | None = None,
    train_category: str | None = None,
):
    """Compatibility loader for the old single-signal PromptAD CLI.

    Unlike the removed implementation, this reads the real cell directories,
    fails loudly when they are absent, and enforces the same non-overlapping
    support/test time split. Formal paper experiments should use the shared
    target-scene manifest instead.
    """

    train_scene = train_category or category
    train_normal_root, _, _ = rf_cell_roots(signal, train_scene, noise_level)
    test_normal_root, test_abnormal_root, gt_root = rf_cell_roots(
        signal, category, noise_level
    )
    _require_directory(train_normal_root, "RF support normal")
    _require_directory(test_normal_root, "RF test normal")
    _require_directory(test_abnormal_root, "RF test abnormal")

    support_paths = _filter_paths(
        train_normal_root.glob("*.png"),
        frequency_band=freq,
        exact_time_range=SUPPORT_TIME_RANGE,
    )
    if k_shot > 0:
        support_paths = support_paths[:k_shot]
    if not support_paths:
        raise RuntimeError(
            f"No RF support samples for {signal}/{train_scene}/{noise_level}, "
            f"time={SUPPORT_TIME_RANGE}, freq={freq!r}"
        )

    normal_test_paths = _filter_paths(
        test_normal_root.glob("*.png"),
        frequency_band=freq,
        min_time_start=TEST_MIN_TIME_START,
    )
    abnormal_test_paths = _filter_paths(
        test_abnormal_root.glob("*.png"),
        frequency_band=freq,
        min_time_start=TEST_MIN_TIME_START,
    )
    if not normal_test_paths or not abnormal_test_paths:
        raise RuntimeError(
            f"Incomplete RF test cell {signal}/{category}/{noise_level}: "
            f"normal={len(normal_test_paths)}, abnormal={len(abnormal_test_paths)}"
        )

    train_data = (
        [str(path) for path in support_paths],
        [0] * len(support_paths),
        [0] * len(support_paths),
        [f"{train_scene}_normal"] * len(support_paths),
    )

    test_paths = [str(path) for path in normal_test_paths]
    test_gts: list[str | int] = [0] * len(normal_test_paths)
    test_labels = [0] * len(normal_test_paths)
    test_types = [f"{category}_normal"] * len(normal_test_paths)
    for path in abnormal_test_paths:
        gt_name = path.name.replace("_abnormal.png", "_groundtruth.png")
        gt_path = gt_root / gt_name
        test_paths.append(str(path))
        test_gts.append(str(gt_path) if gt_path.exists() else 0)
        test_labels.append(1)
        test_types.append(f"{category}_abnormal")

    return train_data, (test_paths, test_gts, test_labels, test_types)
