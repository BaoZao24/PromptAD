"""Run a scene-complete UniVAD texture adaptation check on OFDMA v2.

The formal OFDMA unit is an observation containing 21 sensing-unit (SU)
spectrograms.  This evaluator scores all SU images, takes the maximum score
within each observation, and reports observation-level metrics.  It defaults
to one complete test scene as a resource-bounded first experiment; this is
not presented as the 30-scene macro result.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.ofdma_spectrum import OFDMASpectrogramPreprocessor, NUM_SUS  # noqa: E402
from datasets.ofdma_target_scene import (  # noqa: E402
    DEFAULT_TARGET_SCENE_ROOT,
    build_target_scene_records,
    load_target_scene_manifest,
)

for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        del sys.modules[module_name]

from tools.eval_univad_public_rf import encode_query, score_encoded_multi  # noqa: E402
from tools.eval_univad_rf_fewshot import (  # noqa: E402
    build_model,
    metric_summary,
    setup_texture_memory,
)


DEFAULT_DATA_ROOT = Path("/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic")
DEFAULT_OURS_CSV = ROOT / "analysis_outputs/exploratory/20260815_tta_position_no_tta/ofdma/results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--scene-ids", nargs="+", default=["test_000"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--match-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--dino-model",
        choices=("dinov2_vitg14", "dinov2_vitb14"),
        default="dinov2_vitg14",
    )
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-anomaly-observations-per-type", type=int, default=0)
    parser.add_argument("--ours-csv", default=str(DEFAULT_OURS_CSV))
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def record_tensor(record, preprocessor: OFDMASpectrogramPreprocessor, image_size: int) -> torch.Tensor:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(record.image_path)
    image = preprocessor(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def load_ours(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["target_scene_id"], int(row["shot"])): {
                metric: float(row[metric])
                for metric in ("auroc", "auprc", "fpr95")
            }
            for row in csv.DictReader(handle)
            if row["row_type"] == "target_scene" and row["scope"] == "overall"
        }


def save_progress(args: argparse.Namespace, rows: list[dict], status: str) -> None:
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": status,
                "method": "UniVAD-Texture-adapted",
                "official_method": "UniVAD",
                "dino_model": args.dino_model,
                "dino_backbone": {
                    "dinov2_vitg14": "DINOv2-G/14",
                    "dinov2_vitb14": "DINOv2-B/14",
                }[args.dino_model],
                "protocol_note": "OFDMA v2-realistic; observation score=max over 21 SUs; whole-image texture branch; C3/CAPM/GECM omitted",
                "scene_count_completed": len({row["target_scene_id"] for row in rows}),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    data_root = Path(args.data_root).resolve()
    manifest = load_target_scene_manifest(data_root)
    preprocessor = OFDMASpectrogramPreprocessor(
        data_root,
        output_size=240,
        subcarriers_per_rb=12,
        geometry="letterbox",
        # Frozen by the formal OFDMA protocol; never estimate from test rows.
        min_db=-183.38843,
        max_db=27.147293,
    )
    ours = load_ours(Path(args.ours_csv))
    model = build_model(args)
    rows: list[dict] = []

    for scene_id in args.scene_ids:
        support_records = build_target_scene_records(
            data_root,
            manifest,
            scene_id,
            split="test",
            role="support",
            shot=4,
        )
        test_records = build_target_scene_records(
            data_root,
            manifest,
            scene_id,
            split="test",
            role="test",
            max_normal_observations=args.max_normal_observations,
            max_anomaly_observations_per_type=args.max_anomaly_observations_per_type,
        )
        if len(support_records) != 4 * NUM_SUS:
            raise ValueError(f"{scene_id}: expected 84 support frames, got {len(support_records)}")
        print(
            f"[scene {scene_id}] support_obs=4 test_frames={len(test_records)} "
            f"test_obs={len(test_records) // NUM_SUS}",
            flush=True,
        )
        support_images = torch.stack(
            [record_tensor(record, preprocessor, args.image_size) for record in support_records]
        )
        setup_texture_memory(model, support_images, args.encode_batch_size)
        memory = (
            model.normal_image_features,
            model.normal_patch_tokens,
            model.normal_dino_patches,
        )
        support_indices_by_k = {
            k: torch.arange(k * NUM_SUS, device=model.device, dtype=torch.long)
            for k in (1, 2, 4)
        }
        frame_scores = {k: [] for k in (1, 2, 4)}
        frame_labels = []
        for start in range(0, len(test_records), args.batch_size):
            batch_records = test_records[start : start + args.batch_size]
            images = torch.stack(
                [record_tensor(record, preprocessor, args.image_size) for record in batch_records]
            )
            with torch.inference_mode():
                encoded = encode_query(model, images)
                scores = score_encoded_multi(
                    model,
                    encoded,
                    memory,
                    support_indices_by_k,
                    args.match_chunk_size,
                )
            for k, values in scores.items():
                frame_scores[k].extend(float(value) for value in values.detach().cpu())
            frame_labels.extend(int(record.label) for record in batch_records)
            if (start // args.batch_size + 1) % 100 == 0:
                print(f"  frames={min(start + args.batch_size, len(test_records))}/{len(test_records)}", flush=True)

        for k in (1, 2, 4):
            observation_labels = []
            observation_scores = []
            for start in range(0, len(test_records), NUM_SUS):
                labels = frame_labels[start : start + NUM_SUS]
                scores = frame_scores[k][start : start + NUM_SUS]
                if len(labels) != NUM_SUS or len(scores) != NUM_SUS:
                    raise ValueError(f"{scene_id}: malformed 21-SU observation at frame {start}")
                if len(set(labels)) != 1:
                    raise ValueError(f"{scene_id}: inconsistent labels within observation")
                observation_labels.append(labels[0])
                observation_scores.append(max(scores))
            metrics = metric_summary(observation_labels, observation_scores)
            rows.append(
                {
                    "method": "UniVAD-Texture-adapted",
                    "official_method": "UniVAD",
                    "target_scene_id": scene_id,
                    "shot": k,
                    "support_observations": k,
                    "test_observations": len(observation_labels),
                    "metrics": metrics,
                    "ours_existing_metrics_same_scene": ours.get((scene_id, k)),
                }
            )
        save_progress(args, rows, "running")
        del support_images, memory, frame_scores, frame_labels
        gc.collect()
        if model.device.type == "cuda":
            torch.cuda.empty_cache()

    final = {
        "status": "complete",
        "method": "UniVAD-Texture-adapted",
        "official_method": "UniVAD",
        "dino_model": args.dino_model,
        "dino_backbone": {
            "dinov2_vitg14": "DINOv2-G/14",
            "dinov2_vitb14": "DINOv2-B/14",
        }[args.dino_model],
        "protocol_note": "OFDMA v2-realistic; observation score=max over 21 SUs; whole-image texture branch; C3/CAPM/GECM omitted",
        "scene_count": len(args.scene_ids),
        "rows": rows,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "match_chunk_size": args.match_chunk_size,
    }
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    Path(args.output_json).write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
