#!/usr/bin/env python3
"""Write the shared public-RF support selection used by repeated runs.

The manifest is intentionally model-independent.  ViT and CNN evaluators read
the same deterministic selection rule, while this file makes the exact paths
and their digest auditable for every support replicate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.public_rf_support import build_manifest


def manifest(args: argparse.Namespace) -> dict:
    return build_manifest(seed=args.seed, normal_sampling=args.normal_sampling)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--normal-sampling", default="per_frequency")
    args = parser.parse_args()
    value = manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
