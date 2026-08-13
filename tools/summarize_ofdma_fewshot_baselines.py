#!/usr/bin/env python
"""Build one scene-level OFDMA few-shot table from saved baseline scores."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ofdma_spectrum import (
    DEFAULT_OFDMA_ROOT,
    JAMMER_TYPES,
    NO_JAMMER,
    load_ofdma_labels,
)


SCENE_PATTERN = re.compile(r"spectrogram-(\d{5})-(\d{2})\.png$")


def safe_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else float("nan"),
    }


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=OUTPUT_ROOT, got: {value}")
        method, root = value.split("=", 1)
        if not method or not root:
            raise ValueError(f"Expected METHOD=OUTPUT_ROOT, got: {value}")
        parsed.append((method, Path(root)))
    return parsed


def scene_ids_from_paths(paths) -> np.ndarray:
    scene_ids = []
    for path in paths:
        match = SCENE_PATTERN.search(Path(str(path)).name)
        if match is None:
            raise ValueError(f"Cannot parse OFDMA scene id from: {path}")
        scene_ids.append(int(match.group(1)))
    return np.asarray(scene_ids, dtype=np.int32)


def aggregate(scene_ids, labels, jammer_types, scores, reduction: str):
    grouped = defaultdict(list)
    for index, scene_id in enumerate(scene_ids):
        grouped[int(scene_id)].append(index)

    scene_labels = []
    scene_types = []
    scene_scores = []
    for scene_id in sorted(grouped):
        indices = np.asarray(grouped[scene_id], dtype=np.int64)
        if len(np.unique(labels[indices])) != 1 or len(np.unique(jammer_types[indices])) != 1:
            raise RuntimeError(f"Inconsistent labels inside OFDMA scene {scene_id}")
        selected = scores[indices]
        scene_labels.append(int(labels[indices[0]]))
        scene_types.append(str(jammer_types[indices[0]]))
        scene_scores.append(float(selected.mean() if reduction == "mean" else selected.max()))
    return (
        np.asarray(scene_labels, dtype=np.int32),
        np.asarray(scene_types),
        np.asarray(scene_scores, dtype=np.float32),
    )


def metric_rows(shot, level, method, labels, jammer_types, scores):
    rows = []
    per_type = []
    for scope in ("overall", *JAMMER_TYPES):
        if scope == "overall":
            mask = np.ones(len(labels), dtype=bool)
        else:
            mask = (jammer_types == NO_JAMMER) | (jammer_types == scope)
        metrics = safe_metrics(labels[mask], scores[mask])
        rows.append(
            {
                "shot": shot,
                "level": level,
                "scope": scope,
                "method": method,
                "num_samples": int(mask.sum()),
                **metrics,
            }
        )
        if scope != "overall":
            per_type.append(metrics)
    rows.append(
        {
            "shot": shot,
            "level": level,
            "scope": "macro_jammer",
            "method": method,
            "num_samples": int(len(labels)),
            **{
                key: float(np.nanmean([item[key] for item in per_type]))
                for key in ("auroc", "auprc", "fpr95")
            },
        }
    )
    return rows


def read_existing(path: Path) -> list[dict]:
    if not path:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="METHOD=OUTPUT_ROOT")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--dataset-root", default=str(DEFAULT_OFDMA_ROOT))
    parser.add_argument("--existing-results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/20260718_ofdma_fewshot_baselines/results.csv"),
    )
    args = parser.parse_args()

    labels_df = load_ofdma_labels(args.dataset_root).set_index("scene_id")
    rows = read_existing(args.existing_results) if args.existing_results else []
    for method, root in parse_inputs(args.input):
        for shot in args.shots:
            score_path = root / "scores" / f"ofdma-ofdma-official_split-{shot}shot-scores.npz"
            if not score_path.is_file():
                raise FileNotFoundError(score_path)
            payload = np.load(score_path, allow_pickle=False)
            scores = np.asarray(payload["scores"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int32)
            scene_ids = scene_ids_from_paths(payload["image_paths"])
            jammer_types = np.asarray(
                [str(labels_df.loc[int(scene_id), "jammer_type"]) for scene_id in scene_ids]
            )
            if not (len(scores) == len(labels) == len(scene_ids)):
                raise RuntimeError(f"Length mismatch in {score_path}")

            rows.extend(metric_rows(shot, "image", method, labels, jammer_types, scores))
            for reduction in ("mean", "max"):
                scene_labels, scene_types, scene_scores = aggregate(
                    scene_ids,
                    labels,
                    jammer_types,
                    scores,
                    reduction,
                )
                rows.extend(
                    metric_rows(
                        shot,
                        f"scene_{reduction}",
                        method,
                        scene_labels,
                        scene_types,
                        scene_scores,
                    )
                )

    if not rows:
        raise RuntimeError("No result rows were supplied")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
