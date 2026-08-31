"""Evaluate UniVAD's transparent texture adapter on FedJam spectrograms.

Only the embedded spectrogram image is used.  FedJam's KPI sequence and the
object/component branches of UniVAD are intentionally excluded.  The benign
1/2/4-shot support memories are nested, so every test image is encoded once
and scored against all three memories.
"""

from __future__ import annotations

import argparse
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

# Import the project Arrow helpers before the official UniVAD package is put
# on sys.path.  The UniVAD launcher then uses its own top-level ``utils``
# namespace; no project utils are imported after that point.
from tools.eval_fedjam_fewshot_dual import (  # noqa: E402
    LABEL_NAMES,
    iter_test_records,
    select_benign_support,
)

for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        del sys.modules[module_name]

from tools.eval_univad_public_rf import (  # noqa: E402
    encode_query,
    score_encoded_multi,
)
from tools.eval_univad_rf_fewshot import (  # noqa: E402
    build_model,
    metric_summary,
    setup_texture_memory,
)


DEFAULT_DATA_ROOT = Path("/mnt/data/wangbei/data/FedJam")
DEFAULT_OURS_SUMMARY = ROOT / "analysis_outputs/exploratory/20260815_tta_position_no_tta/fedjam/summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
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
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--ours-summary", default=str(DEFAULT_OURS_SUMMARY))
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def record_tensor(record, image_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(record.image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def load_ours_summary(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for shot in (1, 2, 4):
        value = payload.get(f"shot_{shot}", {}).get("confidence_gated_dual_visual")
        if value:
            result[str(shot)] = {
                "auroc": float(value["auroc"]),
                "auprc": float(value["auprc"]),
                "fpr95": float(value["fpr95"]),
            }
    return result


def save_progress(args: argparse.Namespace, status: str, processed: int, rows: list[dict]) -> None:
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "method": "UniVAD-Texture-adapted",
        "official_method": "UniVAD",
        "dino_model": args.dino_model,
        "dino_backbone": {
            "dinov2_vitg14": "DINOv2-G/14",
            "dinov2_vitb14": "DINOv2-B/14",
        }[args.dino_model],
        "protocol_note": "FedJam full independent test; spectrogram only; whole-image texture branch; C3/CAPM/GECM omitted",
        "support_protocol": "nested benign-only 1/2/4-shot",
        "processed_test_rows": processed,
        "test_total_expected": 7200 if args.max_test_per_label == 0 else None,
        "rows": rows,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "match_chunk_size": args.match_chunk_size,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")
    data_root = Path(args.data_root).resolve()
    print(f"[univad-fedjam] data_root={data_root} batch={args.batch_size}", flush=True)

    support_pool, train_counts, benign_seen = select_benign_support(
        data_root, max_shot=4, seed=args.seed
    )
    support_images = torch.stack(
        [record_tensor(record, args.image_size) for record in support_pool]
    )
    model = build_model(args)
    setup_texture_memory(model, support_images, args.encode_batch_size)
    memory = (
        model.normal_image_features,
        model.normal_patch_tokens,
        model.normal_dino_patches,
    )
    support_indices_by_k = {
        k: torch.arange(k, device=model.device, dtype=torch.long)
        for k in (1, 2, 4)
    }
    ours = load_ours_summary(Path(args.ours_summary))
    scores_by_k = {k: [] for k in (1, 2, 4)}
    labels: list[int] = []
    processed = 0
    batch_records: list = []
    rows: list[dict] = []

    def flush_batch(records: list) -> None:
        nonlocal processed
        if not records:
            return
        images = torch.stack([record_tensor(record, args.image_size) for record in records])
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
            scores_by_k[k].extend(float(value) for value in values.detach().cpu())
        labels.extend(int(record.label != 0) for record in records)
        processed += len(records)
        if processed % (args.batch_size * 100) < len(records):
            print(f"  processed={processed}/{'7200' if args.max_test_per_label == 0 else '?'}", flush=True)
        if processed % (args.batch_size * 200) < len(records):
            save_progress(args, "running", processed, rows)
        del images, encoded, scores

    for record in iter_test_records(data_root, args.max_test_per_label):
        batch_records.append(record)
        if len(batch_records) >= args.batch_size:
            flush_batch(batch_records)
            batch_records = []
    flush_batch(batch_records)
    if not labels or len(set(labels)) < 2:
        raise RuntimeError("FedJam test must contain benign and abnormal rows")

    for k in (1, 2, 4):
        metrics = metric_summary(labels, scores_by_k[k])
        rows.append(
            {
                "method": "UniVAD-Texture-adapted",
                "official_method": "UniVAD",
                "shot": k,
                "support_count": k,
                "test_count": len(labels),
                "test_normal_count": int(sum(label == 0 for label in labels)),
                "test_abnormal_count": int(sum(label == 1 for label in labels)),
                "metrics": metrics,
                "ours_existing_metrics_full_test": ours.get(str(k)),
            }
        )
    final = {
        "status": "complete",
        "method": "UniVAD-Texture-adapted",
        "official_method": "UniVAD",
        "dino_model": args.dino_model,
        "dino_backbone": {
            "dinov2_vitg14": "DINOv2-G/14",
            "dinov2_vitb14": "DINOv2-B/14",
        }[args.dino_model],
        "protocol_note": "FedJam full independent test; spectrogram only; whole-image texture branch; C3/CAPM/GECM omitted",
        "support_protocol": "nested benign-only 1/2/4-shot",
        "train_counts": train_counts,
        "benign_support_pool_seen": benign_seen,
        "test_count": len(labels),
        "test_label_counts": {"normal": int(sum(label == 0 for label in labels)), "abnormal": int(sum(label == 1 for label in labels))},
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
