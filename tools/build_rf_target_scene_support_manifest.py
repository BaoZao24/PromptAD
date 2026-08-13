#!/usr/bin/env python
"""Build and validate the self-RF target-scene normal-support manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.rf_target import (  # noqa: E402
    RF_JSR_BY_SIGNAL as JSR_BY_SIGNAL,
    RF_SCENES as SCENES,
    RF_TARGET_SIGNALS as SIGNALS,
    collect_rf_target_samples as collect_samples,
)
from utils.rf_scene_support import (  # noqa: E402
    SCENE_SUPPORT_SAMPLING_CHOICES,
    build_rf_target_scene_manifest,
    write_rf_target_scene_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--normal-sampling",
        choices=SCENE_SUPPORT_SAMPLING_CHOICES,
        default="per_frequency",
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--signals", nargs="+", choices=SIGNALS, default=SIGNALS)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_rf_target_scene_manifest(
        collect_samples=collect_samples,
        signals=args.signals,
        scenes=args.scenes,
        jsr_by_signal=JSR_BY_SIGNAL,
        normal_sampling=args.normal_sampling,
        seed=args.seed,
    )
    output = write_rf_target_scene_manifest(args.output, manifest)
    summary = {
        "output": str(output),
        "manifest_sha256": manifest["manifest_sha256"],
        "normal_sampling": manifest["normal_sampling"],
        "seed": manifest["seed"],
        "time_block_isolation": manifest["time_block_isolation"],
        "scenes": {
            entry["scene"]: {
                "support_count": entry["support_count"],
                "raw_candidate_count": entry["raw_candidate_count"],
                "unique_content_count": entry["unique_content_count"],
                "excluded_test_content_copies": entry["excluded_test_content_copies"],
                "test_normal_count": sum(
                    cell["test_normal_count"]
                    for cell in manifest["cells"]
                    if cell["scene"] == entry["scene"]
                ),
                "test_abnormal_count": sum(
                    cell["test_abnormal_count"]
                    for cell in manifest["cells"]
                    if cell["scene"] == entry["scene"]
                ),
            }
            for entry in manifest["scene_support"]
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
