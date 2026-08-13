#!/usr/bin/env python3
"""Export only the formal ViT/CNN RF score keys used by the paper figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.confidence_gate import confidence_gated_or, safe_support_only_gate


def rf_scene(path: Path) -> str:
    stem = path.stem
    if stem.startswith("in_house_rf-"):
        return stem.split("-", 3)[2]
    return "public_rf"


def load_reference(root: Path, scene: str, key: str) -> np.ndarray:
    filename = "public_rf.npz" if scene == "public_rf" else f"{scene}.npz"
    path = root / "support_reference" / filename
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gate-protocol",
        choices=["support_only", "transductive"],
        default="support_only",
    )
    parser.add_argument("--vit-reference-dir", type=Path)
    parser.add_argument("--cnn-reference-dir", type=Path)
    args = parser.parse_args()

    source = args.input_root / "scores"
    target = args.output_root / "scores"
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No RF score files under {source}")
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            for key in ("paths", "labels", "vit_scores", "cnn_scores"):
                if key not in data.files:
                    raise KeyError(f"{path} missing key: {key}")
            if args.gate_protocol == "support_only":
                if args.vit_reference_dir is None or args.cnn_reference_dir is None:
                    raise ValueError(
                        "support_only requires --vit-reference-dir and "
                        "--cnn-reference-dir"
                    )
                scene = rf_scene(path)
                current_gate = safe_support_only_gate(
                    data["vit_scores"],
                    data["cnn_scores"],
                    load_reference(args.vit_reference_dir, scene, "vit_scores"),
                    load_reference(args.cnn_reference_dir, scene, "cnn_scores"),
                )["score"]
            else:
                current_gate = confidence_gated_or(
                    data["vit_scores"], data["cnn_scores"]
                )["score"]
            np.savez_compressed(
                target / path.name,
                paths=data["paths"],
                labels=data["labels"],
                vit_scores=data["vit_scores"],
                cnn_scores=data["cnn_scores"],
                vit_only=data["vit_scores"],
                confidence_gated_score=current_gate,
                gate_protocol=np.asarray(args.gate_protocol),
            )
    (args.output_root / "README.md").write_text(
        "# RF dual-visual formal scores\n\n"
        "This bundle contains ViT/CNN branch scores and the formal safe "
        "support-only confidence-gated score. Transductive output is retained "
        "only when explicitly requested as a protocol comparison.\n",
        encoding="utf-8",
    )
    print(f"[done] exported {len(paths)} score files to {target}")


if __name__ == "__main__":
    main()
