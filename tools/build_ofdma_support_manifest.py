#!/usr/bin/env python3
"""Record the exact seeded OFDMA support observations for an experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.ofdma_target_scene import (  # noqa: E402
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--scene-ids", nargs="*", default=[])
    args = parser.parse_args()

    manifest = load_target_scene_manifest(args.dataset_root)
    scene_ids = args.scene_ids or available_target_scenes(manifest, args.split)
    rows = []
    for scene_id in scene_ids:
        for shot in sorted(set(args.shots)):
            records = build_target_scene_records(
                args.dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="support",
                shot=shot,
                support_seed=args.seed,
            )
            observations = sorted({record.observation_id for record in records})
            paths = sorted(str(record.image_path) for record in records)
            rows.append(
                {
                    "target_scene_id": scene_id,
                    "shot": shot,
                    "observation_ids": observations,
                    "image_paths": paths,
                }
            )
    payload = {
        "protocol": "ofdma_target_scene_seeded_support",
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "seed": int(args.seed),
        "scenes": scene_ids,
        "shots": sorted(set(args.shots)),
        "support": rows,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "manifest_sha256": payload["manifest_sha256"], "scenes": len(scene_ids)}, indent=2))


if __name__ == "__main__":
    main()
