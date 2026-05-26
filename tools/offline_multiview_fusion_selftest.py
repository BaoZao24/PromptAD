import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from datasets import get_dataloader_from_args
from PromptAD import PromptAD
from utils.training_utils import get_dir_from_args, setup_seed

SCENES = [
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
]
DATASET_NOISES = {
    "burst_signal": ["m10db", "m20db", "m30db"],
    "chirp_signal": ["m10db", "m20db", "m30db"],
    "dsss_signal": ["m10db", "m20db", "m30db"],
    "wideband_pulse": ["m20db", "m30db", "m40db"],
}
VIEW_CONFIGS = {
    "rgb": {
        "input_mode": "rgb",
        "roots": {
            "burst_signal": "archive_results/archived_result_dirs/rf_rgb",
            "chirp_signal": "archive_results/archived_result_dirs/rf_rgb",
            "dsss_signal": "archive_results/archived_result_dirs/rf_rgb",
            "wideband_pulse": "archive_results/archived_result_dirs/result_wideband_pulse/rf_rgb",
        },
    },
    "spectral_gradient_v2": {
        "input_mode": "spectral_gradient_v2",
        "roots": {
            "burst_signal": "archive_results/archived_result_dirs/rf_grad_v2_selftest",
            "chirp_signal": "archive_results/archived_result_dirs/rf_grad_v2_selftest",
            "dsss_signal": "result_split_ablation/rf_grad_v2_multiview_selftest",
            "wideband_pulse": "result_split_ablation/rf_grad_v2_multiview_selftest",
        },
    },
    "dsss_weak_residual": {
        "input_mode": "dsss_weak_residual",
        "roots": {
            "burst_signal": "result_split_ablation/rf_dsss_weak_residual_multiview_selftest",
            "chirp_signal": "result_split_ablation/rf_dsss_weak_residual_multiview_selftest",
            "dsss_signal": "archive_results/archived_result_dirs/rf_dsss_weak_residual",
            "wideband_pulse": "result_split_ablation/rf_dsss_weak_residual_multiview_selftest",
        },
    },
}


def base_kwargs(dataset, scene, noise_level, input_mode, root_dir, seed):
    return {
        "dataset": dataset,
        "class_name": scene,
        "img_resize": 240,
        "img_cropsize": 240,
        "resolution": 400,
        "batch_size": 400,
        "vis": False,
        "root_dir": root_dir,
        "load_memory": True,
        "cal_pro": False,
        "seed": seed,
        "gpu_id": 0,
        "pure_test": False,
        "k_shot": 1,
        "backbone": "ViT-B-16-plus-240",
        "pretrained_dataset": "laion400m_e32",
        "use_cpu": 0,
        "n_ctx": 4,
        "n_ctx_ab": 1,
        "n_pro": 3,
        "n_pro_ab": 4,
        "Epoch": 50,
        "prompt_mode": "rf",
        "input_mode": input_mode,
        "cls_score_mode": "text_only",
        "visual_topk_ratio": 0.05,
        "visual_score_alpha": 1.0,
        "visual_score_beta": 1.0,
        "visual_score_gamma": 0.0,
        "visual_freq_position_weight": 0.0,
        "stat_fusion": False,
        "stat_topk_ratio": 0.05,
        "stat_fusion_beta": 0.5,
        "visual_adapter": False,
        "adapter_bottleneck_ratio": 0.25,
        "adapter_alpha": 0.2,
        "visual_lora": False,
        "visual_lora_rank": 4,
        "visual_lora_alpha": 8.0,
        "visual_lora_dropout": 0.0,
        "split_mode": "normal_75_25",
        "normal_train_ratio": 0.75,
        "noise_level": noise_level,
        "lr": 0.002,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lambda1": 0.001,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "out_size_h": 400,
        "out_size_w": 400,
    }


def cache_file(cache_dir, view, dataset, scene, noise):
    return cache_dir / view / dataset / scene / f"{noise}.npz"


def score_view(kwargs, checkpoint_path, device):
    setup_seed(kwargs["seed"])
    test_loader, _ = get_dataloader_from_args(phase="test", **kwargs)
    model = PromptAD(**kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    feature_gallery1 = state.pop("feature_gallery1", None)
    feature_gallery2 = state.pop("feature_gallery2", None)
    has_prompt_state = any(k.startswith("prompt_learner.") for k in state.keys())
    model.load_state_dict(state, strict=False)
    if has_prompt_state:
        model.build_text_feature_gallery()
    if feature_gallery1 is not None:
        model.feature_gallery1 = feature_gallery1.to(device)
    if feature_gallery2 is not None:
        model.feature_gallery2 = feature_gallery2.to(device)
    model.eval_mode()
    scores = []
    labels = []
    names = []
    with torch.no_grad():
        for raw_data, mask, label, name, img_type in test_loader:
            data_t = [model.transform(Image.fromarray(f.numpy())) for f in raw_data]
            data_t = torch.stack(data_t, dim=0).to(device)
            vf_gpu = model.encode_image(data_t)
            score_img, _ = model.score_cached(vf_gpu, "cls")
            scores.extend(float(x) for x in score_img)
            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32), np.asarray(names)


def load_or_score(cache_dir, view, dataset, scene, noise, seed, device):
    cfg = VIEW_CONFIGS[view]
    root_dir = cfg["roots"][dataset]
    kwargs = base_kwargs(dataset, scene, noise, cfg["input_mode"], root_dir, seed)
    out = cache_file(cache_dir, view, dataset, scene, noise)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        saved = np.load(out, allow_pickle=True)
        return saved["scores"], saved["labels"], saved["names"]

    _, _, checkpoint_path = get_dir_from_args("CLS", **kwargs)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {view} {dataset} {scene} {noise}: {checkpoint_path}")
    scores, labels, names = score_view(kwargs, checkpoint_path, device)
    np.savez_compressed(out, scores=scores, labels=labels, names=names)
    return scores, labels, names


def conservative_fusion(rgb, aux_best, lam):
    return rgb + lam * np.maximum(aux_best - rgb, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--lambda-conservative", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="analysis_outputs/multiview_fusion_selftest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "score_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rows = []
    for dataset, noise_levels in DATASET_NOISES.items():
        for scene in SCENES:
            for noise in noise_levels:
                view_scores = {}
                base_labels = None
                base_names = None
                for view in VIEW_CONFIGS:
                    scores, labels, names = load_or_score(cache_dir, view, dataset, scene, noise, args.seed, device)
                    if base_labels is None:
                        base_labels = labels
                        base_names = names
                    else:
                        if not np.array_equal(labels, base_labels):
                            raise ValueError(f"Label mismatch for {dataset} {scene} {noise} view={view}")
                        if not np.array_equal(names, base_names):
                            raise ValueError(f"Name mismatch for {dataset} {scene} {noise} view={view}")
                    view_scores[view] = scores

                rgb = view_scores["rgb"]
                grad = view_scores["spectral_gradient_v2"]
                weak = view_scores["dsss_weak_residual"]
                aux_best = np.maximum(grad, weak)
                fused = {
                    "rgb": rgb,
                    "max": np.maximum(np.maximum(rgb, grad), weak),
                    "mean": (rgb + grad + weak) / 3.0,
                    "conservative_lam0.5": conservative_fusion(rgb, aux_best, args.lambda_conservative),
                }
                result = {
                    "dataset": dataset,
                    "scene": scene,
                    "noise": noise,
                }
                for rule_name, scores in fused.items():
                    result[rule_name] = float(roc_auc_score(base_labels, scores) * 100.0)
                rows.append(result)

    detail_df = pd.DataFrame(rows).sort_values(["dataset", "scene", "noise"]).reset_index(drop=True)
    detail_path = output_dir / "fusion_detail.csv"
    detail_df.to_csv(detail_path, index=False, float_format="%.4f")

    rule_cols = ["rgb", "max", "mean", "conservative_lam0.5"]
    dataset_summary = detail_df.groupby("dataset")[rule_cols].mean().reset_index()
    overall = {"dataset": "OVERALL"}
    for col in rule_cols:
        overall[col] = detail_df[col].mean()
    dataset_summary = pd.concat([dataset_summary, pd.DataFrame([overall])], ignore_index=True)
    summary_path = output_dir / "fusion_summary.csv"
    dataset_summary.to_csv(summary_path, index=False, float_format="%.4f")

    overall_row = dataset_summary[dataset_summary["dataset"] == "OVERALL"].iloc[0]
    best_rule = max(rule_cols, key=lambda c: float(overall_row[c]))
    best_path = output_dir / "best_rule.txt"
    best_path.write_text(best_rule + "\n")

    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")
    print(f"best_rule={best_rule}")
    print(dataset_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
