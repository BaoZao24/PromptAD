"""Small, frozen-feature adapters for candidate anomaly methods.

This script is deliberately separate from the formal evaluator.  It lets us
compare the core ideas of SubspaceAD, RAD, and RareCLIP on the same CLIP patch
features and target-scene manifest before changing the main method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PromptAD import PromptAD
from tools.eval_cls_vit_patchcore_gallery import prepare_patch_features
from train_rf_target_pooled_universal import to_model_input


DEFAULT_MANIFEST = ROOT / "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json"
DEFAULT_BASELINE_ROOT = ROOT / "analysis_outputs/exploratory/20260813_frequency_response_memory_rf_full/no_tta_merged/scores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--signal", default="burst_signal")
    parser.add_argument("--scene", default="WeaponMuseum_spectrum")
    parser.add_argument("--jsr", default="m10db")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=("all", "subspace", "rad", "rarity"), default="all")
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--pca-variance", type=float, default=0.70)
    parser.add_argument("--k-image", type=int, default=4)
    parser.add_argument("--position-radius", type=int, default=-1)
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--density-topk", type=int, default=10)
    parser.add_argument("--rarity-alpha", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    return parser.parse_args()


def build_model(device: str) -> PromptAD:
    model = PromptAD(
        out_size_h=400,
        out_size_w=400,
        device=device,
        backbone="ViT-B-16-plus-240",
        pretrained_dataset="laion400m_e32",
        n_ctx=4,
        n_pro=3,
        n_ctx_ab=1,
        n_pro_ab=4,
        class_name="signal",
        precision="fp16",
        k_shot=1,
        img_resize=240,
        img_cropsize=240,
        dataset="rf_target_test_pool",
        prompt_mode="rf",
        input_mode="rgb",
        text_prototype_mode="single",
    ).to(device)
    model.eval()
    return model


def cache_key(path: str) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def encode_path(model: PromptAD, path: str, device: str, cache_dir: Path | None):
    cache_path = cache_dir / f"{cache_key(path)}.npz" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return data["patches"], data["global"]

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    raw = torch.from_numpy(image).unsqueeze(0)
    with torch.no_grad():
        model_input = to_model_input(model, raw, device, rgb_from_bgr=True)
        visual = model.encode_image(model_input)
        patches = prepare_patch_features(visual)[0].float().cpu().numpy()
        global_feature = F.normalize(visual[0].float(), dim=-1)[0].cpu().numpy()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, patches=patches, **{"global": global_feature})
    return patches, global_feature


def get_cell(manifest: dict, signal: str, scene: str, jsr: str) -> tuple[list[dict], dict]:
    scene_entries = [entry for entry in manifest["scene_support"] if entry["scene"] == scene]
    cells = [
        cell
        for cell in manifest["cells"]
        if cell["signal"] == signal and cell["scene"] == scene and cell["jsr"] == jsr
    ]
    if len(scene_entries) != 1 or len(cells) != 1:
        raise KeyError(f"Cannot find unique cell {signal}/{scene}/{jsr}")
    return scene_entries[0]["support"], cells[0]


def image_score(patch_scores: np.ndarray) -> float:
    return float(np.max(patch_scores))


def rad_scores(
    query_patches: np.ndarray,
    query_global: np.ndarray,
    support_patches: list[np.ndarray],
    support_globals: np.ndarray,
    k_image: int,
    position_radius: int,
) -> float:
    similarities = support_globals @ query_global
    if k_image < 0:
        selected = np.arange(len(support_patches))
    else:
        k = max(1, min(int(k_image), len(support_patches)))
        selected = np.argpartition(similarities, -k)[-k:]

    grid_h = int(round(query_patches.shape[0] ** 0.5))
    if grid_h * grid_h != query_patches.shape[0]:
        raise ValueError(f"Expected square patch grid, got {query_patches.shape}")
    grid_w = grid_h

    if position_radius < 0:
        candidates = np.concatenate([support_patches[index] for index in selected], axis=0)
        sims = query_patches @ candidates.T
        return image_score((1.0 - np.max(sims, axis=1)) / 2.0)

    patch_scores = []
    for patch_index, query_patch in enumerate(query_patches):
        row, col = divmod(patch_index, grid_w)
        indices = [
            rr * grid_w + cc
            for rr in range(max(0, row - position_radius), min(grid_h, row + position_radius + 1))
            for cc in range(max(0, col - position_radius), min(grid_w, col + position_radius + 1))
        ]
        candidates = np.concatenate([support_patches[index][indices] for index in selected], axis=0)
        patch_scores.append((1.0 - np.max(query_patch @ candidates.T)) / 2.0)
    return image_score(np.asarray(patch_scores, dtype=np.float32))


def fit_subspace(support: np.ndarray, variance: float) -> tuple[np.ndarray, np.ndarray, int]:
    mean = support.mean(axis=0)
    centered = support - mean
    covariance = centered.T @ centered / max(1, len(support) - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = components[:, order]
    cumulative = np.cumsum(eigenvalues) / max(float(eigenvalues.sum()), 1e-12)
    dimension = int(np.searchsorted(cumulative, variance)) + 1
    dimension = max(1, min(dimension, components.shape[1]))
    return mean, components[:, :dimension], dimension


def subspace_score(query: np.ndarray, mean: np.ndarray, components: np.ndarray) -> float:
    centered = query - mean
    projected = centered @ components
    residual = centered - projected @ components.T
    return image_score(np.linalg.norm(residual, axis=1))


def fit_rarity_memory(support: np.ndarray, density_topk: int) -> tuple[np.ndarray, np.ndarray]:
    similarities = support @ support.T
    np.fill_diagonal(similarities, -1.0)
    k = max(1, min(int(density_topk), max(1, support.shape[0] - 1)))
    density = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
    rarity = 1.0 - density
    rarity = (rarity - rarity.min()) / max(float(rarity.max() - rarity.min()), 1e-6)
    return support, rarity.astype(np.float32)


def rarity_score(
    query: np.ndarray,
    memory: np.ndarray,
    rarity: np.ndarray,
    nn_topk: int,
    alpha: float,
) -> float:
    similarities = query @ memory.T
    k = max(1, min(int(nn_topk), memory.shape[0]))
    top_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
    top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
    distance = (1.0 - top_similarities.mean(axis=1)) / 2.0
    weights = np.exp((top_similarities - top_similarities.max(axis=1, keepdims=True)) / 0.05)
    weights /= weights.sum(axis=1, keepdims=True)
    local_rarity = (weights * rarity[top_indices]).sum(axis=1)
    return image_score(distance * (1.0 + float(alpha) * local_rarity))


def formal_baseline_auc(root: Path, signal: str, scene: str, jsr: str) -> float | None:
    path = root / f"{signal}-{scene}-{jsr}-scores.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return float(roc_auc_score(data["labels"], data["vit_patchcore_max_scores"]) * 100.0)


def evaluate_cell(model: PromptAD, manifest: dict, cell: dict, args: argparse.Namespace, cache_dir: Path | None) -> dict:
    scene_entries = [entry for entry in manifest["scene_support"] if entry["scene"] == cell["scene"]]
    if len(scene_entries) != 1:
        raise KeyError(f"Cannot find scene support for {cell['scene']}")
    support_items = scene_entries[0]["support"]
    query_items = [(item, 0) for item in cell["test_normals"]]
    query_items += [(item, 1) for item in cell["test_abnormals"]]

    support_features = []
    support_globals = []
    for item in support_items:
        patches, global_feature = encode_path(model, item["path"], args.device, cache_dir)
        support_features.append(patches)
        support_globals.append(global_feature)
    support_globals_np = np.asarray(support_globals, dtype=np.float32)
    support_globals_np /= np.linalg.norm(support_globals_np, axis=1, keepdims=True) + 1e-8
    support_matrix = np.concatenate(support_features, axis=0).astype(np.float32)

    query_features = []
    labels = []
    for item, label in query_items:
        query_features.append(encode_path(model, item["path"], args.device, cache_dir))
        labels.append(label)

    baseline = formal_baseline_auc(Path(args.baseline_root), cell["signal"], cell["scene"], cell["jsr"])
    result = {
        "signal": cell["signal"],
        "scene": cell["scene"],
        "jsr": cell["jsr"],
        "baseline": baseline,
    }
    print(f"cell={cell['signal']}/{cell['scene']}/{cell['jsr']} support={len(support_features)} query={len(query_features)}")

    if args.method in ("all", "subspace"):
        mean, components, dimension = fit_subspace(support_matrix, args.pca_variance)
        scores = [subspace_score(patches, mean, components) for patches, _ in query_features]
        result["subspace"] = float(roc_auc_score(labels, scores) * 100.0)
        print(f"subspace variance={args.pca_variance:.3f} components={dimension} auc={result['subspace']:.4f}")

    if args.method in ("all", "rad"):
        for radius in (args.position_radius,):
            scores = [
                rad_scores(
                    patches,
                    global_feature,
                    support_features,
                    support_globals_np,
                    args.k_image,
                    radius,
                )
                for patches, global_feature in query_features
            ]
            result["rad"] = float(roc_auc_score(labels, scores) * 100.0)
            print(f"rad k_image={args.k_image} position_radius={radius} auc={result['rad']:.4f}")

    if args.method in ("all", "rarity"):
        memory, rarity = fit_rarity_memory(support_matrix, args.density_topk)
        scores = [
            rarity_score(patches, memory, rarity, args.nn_topk, args.rarity_alpha)
            for patches, _ in query_features
        ]
        result["rarity"] = float(roc_auc_score(labels, scores) * 100.0)
        print(
            f"rarity density_topk={args.density_topk} nn_topk={args.nn_topk} "
            f"alpha={args.rarity_alpha:.3f} auc={result['rarity']:.4f}"
        )
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    if args.all_cells:
        cells = manifest["cells"]
    else:
        _, selected = get_cell(manifest, args.signal, args.scene, args.jsr)
        cells = [selected]

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    model = build_model(args.device)
    results = [evaluate_cell(model, manifest, cell, args, cache_dir) for cell in cells]
    if args.all_cells:
        for key in ("baseline", "subspace", "rad", "rarity"):
            values = [row[key] for row in results if row.get(key) is not None]
            if values:
                print(f"macro_{key}={np.mean(values):.4f} n={len(values)}")


if __name__ == "__main__":
    main()
