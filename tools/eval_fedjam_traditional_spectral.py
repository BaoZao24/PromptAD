#!/usr/bin/env python
"""Few-shot spectrogram-domain communication baselines on FedJam.

FedJam stores the image column inside local Hugging Face Arrow shards.  This
runner keeps the traditional baselines independent from the visual encoder:
only benign train rows are used for support calibration, while the complete
test split is scored image by image.

The methods are image-domain adaptations of ED, power-spectrum entropy,
spectral flatness, spectral kurtosis, CA-CFAR, and their fixed support-only
statistical fusion.  They should not be described as full IQ-domain
implementations because FedJam's local experiment input is a PNG spectrogram.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.ipc as pa_ipc
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.traditional_spectral import (
    METHOD_DISPLAY_NAMES,
    METHOD_REFERENCES,
    TRADITIONAL_METHODS,
    extract_raw_statistics,
    fit_support_calibrator,
)


LABEL_NAMES = {
    0: "benign",
    1: "pulse",
    2: "single_tone",
    3: "wideband",
}


def image_bytes(image_value: dict) -> bytes:
    if not isinstance(image_value, dict) or not image_value.get("bytes"):
        raise ValueError("FedJam row has no embedded image bytes")
    return bytes(image_value["bytes"])


def decode_image(encoded: bytes) -> np.ndarray:
    with Image.open(BytesIO(encoded)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (224, 224):
        # Keep the same geometry normalization as the visual FedJam runner.
        image = Image.fromarray(rgb, mode="RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(rgb)


def iter_arrow_rows(split_dir: Path):
    files = sorted(split_dir.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards under {split_dir}")
    for shard in files:
        with shard.open("rb") as handle:
            reader = pa_ipc.open_stream(handle)
            for batch_index, batch in enumerate(reader):
                images = batch.column("image").to_pylist()
                labels = batch.column("label").to_pylist()
                for row_index, (image, label) in enumerate(zip(images, labels)):
                    yield (
                        image_bytes(image),
                        int(label),
                        f"{shard.name}:batch{batch_index}:row{row_index}",
                    )


def select_benign_support(data_root: Path, max_shot: int, seed: int):
    """Select a deterministic nested pool from train label=0 only."""

    rng = np.random.default_rng(seed)
    reservoir: list[tuple[bytes, int, str]] = []
    benign_seen = 0
    counts = {label: 0 for label in LABEL_NAMES}
    train_dir = data_root / "fedjam_dataset" / "train"
    for encoded, label, name in iter_arrow_rows(train_dir):
        counts[label] = counts.get(label, 0) + 1
        if label != 0:
            continue
        benign_seen += 1
        item = (encoded, label, name)
        if len(reservoir) < max_shot:
            reservoir.append(item)
            continue
        replacement = int(rng.integers(0, benign_seen))
        if replacement < max_shot:
            reservoir[replacement] = item
    if len(reservoir) < max_shot:
        raise RuntimeError(
            f"FedJam benign support has only {len(reservoir)} rows; "
            f"requested {max_shot}"
        )
    support = [
        {
            "image": decode_image(encoded),
            "label": label,
            "name": name,
            "encoded": encoded,
        }
        for encoded, label, name in reservoir
    ]
    return support, counts, benign_seen


def iter_test_records(data_root: Path, max_per_label: int = 0):
    counts = {label: 0 for label in LABEL_NAMES}
    test_dir = data_root / "fedjam_dataset" / "test"
    for encoded, label, name in iter_arrow_rows(test_dir):
        if max_per_label > 0 and counts.get(label, 0) >= max_per_label:
            continue
        counts[label] = counts.get(label, 0) + 1
        yield {
            "image": decode_image(encoded),
            "label": label,
            "name": name,
        }


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reached[0]] * 100.0) if reached.size else 100.0
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": fpr95,
    }


def append_metric_rows(
    rows: list[dict],
    shot: int,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    scopes: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "overall",
            np.ones(labels.shape, dtype=bool),
            (labels != 0).astype(np.int32),
        )
    ]
    for label_id, name in LABEL_NAMES.items():
        if label_id == 0:
            continue
        mask = (labels == 0) | (labels == label_id)
        scopes.append((name, mask, (labels[mask] == label_id).astype(np.int32)))

    attack_metrics = []
    for scope, mask, binary in scopes:
        values = metric(binary, scores[mask])
        rows.append(
            {
                "shot": int(shot),
                "method": method,
                "method_display": METHOD_DISPLAY_NAMES[method],
                "scope": scope,
                "n": int(mask.sum()),
                "num_normal": int((labels[mask] == 0).sum()),
                "num_abnormal": int(binary.sum()),
                **values,
            }
        )
        if scope != "overall":
            attack_metrics.append(values)
    rows.append(
        {
            "shot": int(shot),
            "method": method,
            "method_display": METHOD_DISPLAY_NAMES[method],
            "scope": "macro_attack",
            "n": int(labels.size),
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels != 0).sum()),
            "auroc": float(np.nanmean([item["auroc"] for item in attack_metrics])),
            "auprc": float(np.nanmean([item["auprc"] for item in attack_metrics])),
            "fpr95": float(np.nanmean([item["fpr95"] for item in attack_metrics])),
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument(
        "--output-root", default="analysis_outputs/20260810_fedjam_traditional_fewshot"
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--frequency-axis",
        type=int,
        choices=[0, 1],
        default=1,
        help="0=vertical frequency, 1=horizontal frequency; FedJam default follows visual TTA convention.",
    )
    args = parser.parse_args()

    shots = sorted(set(int(shot) for shot in args.shots))
    if not shots or min(shots) <= 0:
        raise ValueError("--shots must contain positive integers")
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    support, train_counts, benign_seen = select_benign_support(
        data_root, max(shots), args.seed
    )
    support_manifest = {
        "seed": args.seed,
        "shots": shots,
        "train_counts": train_counts,
        "benign_seen": benign_seen,
        "frequency_axis": args.frequency_axis,
        "support": [
            {
                "name": item["name"],
                "label": item["label"],
                "sha256": hashlib.sha256(item["image"].tobytes()).hexdigest(),
                "shape": list(item["image"].shape),
            }
            for item in support
        ],
    }
    (output_root / "support_manifest.json").write_text(
        json.dumps(support_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[support] selected={len(support)} benign_seen={benign_seen} "
        f"counts={train_counts} frequency_axis={args.frequency_axis}",
        flush=True,
    )

    calibrators = {
        shot: fit_support_calibrator(
            [item["image"] for item in support[:shot]],
            args.frequency_axis,
            TRADITIONAL_METHODS,
        )
        for shot in shots
    }
    score_bank = {
        shot: {method: [] for method in TRADITIONAL_METHODS} for shot in shots
    }
    labels = []
    names = []
    test_counts = {label: 0 for label in LABEL_NAMES}
    test_records = iter_test_records(data_root, args.max_test_per_label)
    for index, record in enumerate(test_records, 1):
        labels.append(record["label"])
        names.append(record["name"])
        test_counts[record["label"]] = test_counts.get(record["label"], 0) + 1
        raw = extract_raw_statistics(record["image"], args.frequency_axis)
        for shot in shots:
            scored = calibrators[shot].score_raw(raw)
            for method in TRADITIONAL_METHODS:
                score_bank[shot][method].append(float(scored[method]))
        if args.progress_every and index % args.progress_every == 0:
            print(f"[test] processed={index} counts={test_counts}", flush=True)

    labels_np = np.asarray(labels, dtype=np.int32)
    names_np = np.asarray(names)
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        raise RuntimeError("FedJam test subset must contain both benign and abnormal rows")
    print(f"[test] counts={test_counts} total={labels_np.size}", flush=True)

    result_rows: list[dict] = []
    summary = {
        "method": "fedjam_traditional_spectrogram_statistics",
        "data_root": str(data_root),
        "shots": shots,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "support_only": True,
        "test_batch_statistics_used": False,
        "train_abnormal_used": False,
        "input_modality": "spectrogram_image_only",
        "spectrogram_domain_adaptation": True,
        "frequency_axis": args.frequency_axis,
        "frequency_axis_convention": (
            "horizontal frequency / vertical time, following the FedJam visual TTA convention"
            if args.frequency_axis == 1
            else "vertical frequency / horizontal time"
        ),
        "label_names": LABEL_NAMES,
        "methods": {
            method: {
                "display": METHOD_DISPLAY_NAMES[method],
                **METHOD_REFERENCES[method],
            }
            for method in TRADITIONAL_METHODS
        },
        "seed": args.seed,
    }
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        arrays = {
            method: np.asarray(score_bank[shot][method], dtype=np.float32)
            for method in TRADITIONAL_METHODS
        }
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_scores.npz",
            labels=labels_np,
            names=names_np,
            **arrays,
        )
        for method in TRADITIONAL_METHODS:
            append_metric_rows(result_rows, shot, method, labels_np, arrays[method])
        summary[f"shot_{shot}"] = {
            method: metric((labels_np != 0).astype(np.int32), arrays[method])
            for method in TRADITIONAL_METHODS
        }

    write_csv(output_root / "metrics.csv", result_rows)
    protocol = {
        **summary,
        "support_selection": "seeded reservoir over train label=benign; nested prefix for shots",
        "test_selection": "all official test rows unless max-test-per-label is set",
        "score_files": [str(path.relative_to(output_root)) for path in sorted(score_dir.glob("*.npz"))],
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# FedJam traditional spectrogram-domain baselines",
        "",
        "- support: benign train rows only; test labels are metrics-only",
        "- input: spectrogram images only; FedJam KPI sequences are not used",
        f"- frequency axis: `{args.frequency_axis}` ({summary['frequency_axis_convention']})",
        "",
        "| Shot | Method | AUROC | AUPRC | FPR@95%TPR | Reference |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for shot in shots:
        for method in TRADITIONAL_METHODS:
            values = summary[f"shot_{shot}"][method]
            lines.append(
                f"| {shot} | {METHOD_DISPLAY_NAMES[method]} | "
                f"{values['auroc']:.2f} | {values['auprc']:.2f} | "
                f"{values['fpr95']:.2f} | {METHOD_REFERENCES[method]['short']} |"
            )
    lines.extend(
        [
            "",
            "These are deterministic spectrogram-domain adaptations of the cited "
            "communication statistics, not full IQ-domain detector reimplementations.",
            "See `metrics.csv`, `protocol.json`, `support_manifest.json`, and `scores/` "
            "for the complete record.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_root / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
