"""Small RF smoke evaluator for a UniVAD texture-branch adaptation.

This is not the official UniVAD-G reproduction.  It keeps the official
whole-image matching recipe but uses locally cached CLIP-L/14 and DINOv2-B/14
weights, because the official DINOv2-G checkpoint is about 4.23 GB.  Object
component segmentation and GECM are intentionally omitted: a spectrogram is
treated as one structured signal image.  The output must be named
``UniVAD-Texture-B-adapted`` in comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
UNIVAD_ROOT = ROOT / "references" / "UniVAD"
if str(UNIVAD_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIVAD_ROOT))

from models import clip as open_clip  # noqa: E402


DEFAULT_MANIFEST = ROOT / "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json"
DEFAULT_OURS_CSV = ROOT / "analysis_outputs/20260725_target_scene_self_rf_formal_seed111/self_fusion/per_cell_confidence_gate.csv"


class UniVADTextureB:
    """Whole-image CLIP/DINO patch matching following UniVAD's texture path."""

    def __init__(self, image_size: int, device: str):
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.out_layers = [6, 12, 18, 24]
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336", self.image_size, pretrained="openai"
        )
        self.clip_model.to(self.device).eval()
        self.dino = torch.hub.load(
            str(UNIVAD_ROOT / "models" / "dinov2"),
            "dinov2_vitb14",
            pretrained=True,
            source="local",
        ).to(self.device).eval()
        self.clip_transform = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.dino_transform = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.inference_mode()
    def setup(self, support: torch.Tensor) -> None:
        clip_input = self.clip_transform(support).to(self.device)
        dino_input = self.dino_transform(support).to(self.device)
        image_features, patch_tokens = self.clip_model.encode_image(clip_input, self.out_layers)
        self.normal_image_features = F.normalize(image_features[:, 0, :], dim=-1)
        self.normal_clip_patches = [tokens[:, 1:, :].contiguous() for tokens in patch_tokens]
        self.normal_dino_patches = self.dino.forward_features(dino_input)["x_norm_patchtokens"]

    @torch.inference_mode()
    def score(self, image: torch.Tensor) -> float:
        clip_input = self.clip_transform(image).to(self.device)
        dino_input = self.dino_transform(image).to(self.device)
        image_features, patch_tokens = self.clip_model.encode_image(clip_input, self.out_layers)
        image_features = F.normalize(image_features[:, 0, :], dim=-1)
        global_score = float((1.0 - (image_features @ self.normal_image_features.T).max()).item())

        clip_distances = []
        for layer_index, tokens in enumerate(patch_tokens):
            if layer_index % 2 == 0:
                continue
            query = F.normalize(tokens[:, 1:, :], dim=-1)
            reference = F.normalize(self.normal_clip_patches[layer_index], dim=-1)
            similarity = torch.einsum("bnd,mnd->bmn", query, reference)
            clip_distances.append(1.0 - similarity.max(dim=-1).values)
        clip_score = torch.stack(clip_distances).mean(dim=0).max()

        query_dino = F.normalize(self.dino.forward_features(dino_input)["x_norm_patchtokens"], dim=-1)
        reference_dino = F.normalize(self.normal_dino_patches, dim=-1)
        dino_similarity = torch.einsum("bnd,mnd->bmn", query_dino, reference_dino)
        dino_score = (1.0 - dino_similarity.max(dim=-1).values).max()
        return float((global_score + ((clip_score + dino_score) / 2.0).item()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--signal", default="burst_signal")
    parser.add_argument("--scene", default="WeaponMuseum_spectrum")
    parser.add_argument("--jsr", default="m30db")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-test-normal", type=int, default=8)
    parser.add_argument("--max-test-abnormal", type=int, default=8)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def load_image(path: str, image_size: int) -> torch.Tensor:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def find_cell(manifest: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
    scenes = [row for row in manifest["scene_support"] if row["scene"] == args.scene]
    cells = [
        row
        for row in manifest["cells"]
        if row["signal"] == args.signal and row["scene"] == args.scene and row["jsr"] == args.jsr
    ]
    if len(scenes) != 1 or len(cells) != 1:
        raise KeyError(f"Cannot find unique cell {args.signal}/{args.scene}/{args.jsr}")
    return scenes[0]["support"], cells[0]


def metric_summary(labels: list[int], scores: list[float]) -> dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels_np, scores_np) * 100.0),
        "auprc": float(average_precision_score(labels_np, scores_np) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else 100.0,
    }


def existing_ours_auc(args: argparse.Namespace) -> float | None:
    with Path(DEFAULT_OURS_CSV).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == args.signal and row["scene"] == args.scene and row["jsr"] == args.jsr:
                return float(row["confidence_gated_auc"])
    return None


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    support_items, cell = find_cell(manifest, args)
    normal_items = cell["test_normals"][: args.max_test_normal]
    abnormal_items = cell["test_abnormals"][: args.max_test_abnormal]
    if not normal_items or not abnormal_items:
        raise ValueError("Smoke test needs both normal and abnormal query samples")

    print(
        f"[univad-texture-b] support={len(support_items)} normal={len(normal_items)} "
        f"abnormal={len(abnormal_items)} device={args.device}",
        flush=True,
    )
    model = UniVADTextureB(args.image_size, args.device)
    support = torch.stack([load_image(row["path"], args.image_size) for row in support_items])
    model.setup(support)

    labels: list[int] = []
    scores: list[float] = []
    for label, row in [(0, item) for item in normal_items] + [(1, item) for item in abnormal_items]:
        image = load_image(row["path"], args.image_size).unsqueeze(0)
        labels.append(label)
        scores.append(model.score(image))
    result = {
        "method": "UniVAD-Texture-B-adapted",
        "official_method": "UniVAD",
        "protocol_note": "whole-image texture branch; local DINOv2-B substitute for official DINOv2-G; C3/GECM omitted",
        "signal": args.signal,
        "scene": args.scene,
        "jsr": args.jsr,
        "support_count": len(support_items),
        "test_normal_count": len(normal_items),
        "test_abnormal_count": len(abnormal_items),
        "metrics": metric_summary(labels, scores),
        "ours_existing_auc_same_cell_full_test": existing_ours_auc(args),
        "image_size": args.image_size,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
