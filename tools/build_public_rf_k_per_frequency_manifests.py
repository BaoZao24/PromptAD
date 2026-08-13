#!/usr/bin/env python3
"""Build nested Public-RF manifests with k normal samples per frequency band."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.public_rf_support import build_k_per_frequency_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()

    for k in sorted(set(args.ks)):
        value = build_k_per_frequency_manifest(seed=args.seed, k=k)
        output = args.output_dir / f"k{k}" / "support_manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[done] k={k}: support={value['support_count']} "
            f"({value['frequency_band_count']} bands x {k}), "
            f"test_normal={value['test_normal_count']} -> {output}"
        )


if __name__ == "__main__":
    main()
