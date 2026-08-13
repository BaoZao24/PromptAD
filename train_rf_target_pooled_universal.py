#!/usr/bin/env python
"""Legacy pooled RF PromptAD training entrypoint.

The current formal gated-fusion protocol uses one support manifest per scene.
This script remains for reproducing older pooled PromptAD checkpoints and for
shared image/model helpers imported by evaluation tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.rf_target import (
    RF_JSR_BY_SIGNAL,
    RF_SCENES,
    RF_TARGET_SIGNALS,
    collect_rf_target_samples,
)
from PromptAD import PromptAD, TripletLoss
from PromptAD.ad_prompts import (
    get_grouped_rf_abnormal_prompt_specs,
    get_rf_signal_key,
    is_rf_prompt_class,
)
from utils.eval_utils import specify_resolution
from utils.rf_frequency_sampling import maybe_select_one_per_frequency_band, split_train_test_normals
from utils.metrics import metric_cal_img, metric_cal_pix
from utils.training_utils import setup_seed


SIGNALS = list(RF_TARGET_SIGNALS)
SCENES = list(RF_SCENES)
JSR_BY_SIGNAL = {
    signal: list(jsrs)
    for signal, jsrs in RF_JSR_BY_SIGNAL.items()
}
METHOD_PRESETS = {
    "pooled_rf_rgb": {},
    "pooled_rf_morph_gray_residual": {
        "input_mode": "morph_fusion_gray_residual_a01",
    },
    "pooled_rf_rgb_vcpa": {
        "visual_class_prompt": True,
        "visual_class_token_num": 2,
        "visual_class_prompt_alpha": 0.2,
        "visual_class_prototype_mode": "mean",
        "visual_class_prototype_num": 1,
    },
    "pooled_rf_rgb_visual_adapter": {
        "visual_adapter": True,
        "adapter_bottleneck_ratio": 0.25,
        "adapter_alpha": 0.2,
    },
    "pooled_rf_rgb_object_agnostic": {
        "prompt_mode": "rf_object_agnostic",
    },
    "pooled_rf_rgb_grouped_meanmax": {
        "text_prototype_mode": "grouped_meanmax",
    },
}


class RFPathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt, label, sample_type, name_prefix = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_h, orig_w = img.shape[:2]
        if gt == 0:
            gt_arr = np.zeros([orig_h, orig_w], dtype=np.uint8)
        else:
            gt_color = cv2.imread(gt, cv2.IMREAD_COLOR)
            if gt_color is None:
                gt_arr = np.zeros([orig_h, orig_w], dtype=np.uint8)
            else:
                lower_yellow = np.array([0, 200, 200])
                upper_yellow = np.array([50, 255, 255])
                yellow_mask = cv2.inRange(gt_color, lower_yellow, upper_yellow)
                if np.any(yellow_mask):
                    gt_arr = yellow_mask
                else:
                    gt_gray = cv2.cvtColor(gt_color, cv2.COLOR_BGR2GRAY)
                    gt_arr = ((gt_gray > 0).astype(np.uint8) * 255)

        target_size = min(max(orig_h, orig_w), 1024)
        img = cv2.resize(img, (target_size, target_size))
        gt_arr = cv2.resize(gt_arr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        img_name = f"{name_prefix}-{sample_type}-{os.path.basename(img_path[:-4])}"
        return img, gt_arr, label, img_name, sample_type


def collect_samples(signal, scene, jsr, phase, k_shot=1, exclude_paths=None):
    """Backward-compatible name for the canonical self-RF data accessor."""

    return collect_rf_target_samples(
        signal,
        scene,
        jsr,
        phase,
        k_shot=k_shot,
        exclude_paths=exclude_paths,
    )


def build_train_loader(args):
    samples = []
    full_normal_modes = {"frequency_one_per_band", "split_75_25"}
    support_k_shot = 0 if args.normal_sampling in full_normal_modes else args.k_shot
    for signal in args.signals:
        for scene in SCENES:
            for jsr in JSR_BY_SIGNAL[signal]:
                cell_samples = collect_samples(signal, scene, jsr, "train", k_shot=support_k_shot)
                if args.normal_sampling == "split_75_25":
                    cell_samples, _ = split_train_test_normals(
                        cell_samples,
                        train_ratio=0.75,
                        seed=args.seed,
                        path_getter=lambda sample: sample[0],
                    )
                samples.extend(cell_samples)
    if args.normal_sampling != "split_75_25":
        samples = maybe_select_one_per_frequency_band(
            samples,
            getattr(args, "normal_sampling", "all"),
            path_getter=lambda sample: sample[0],
        )
    if not samples:
        raise RuntimeError("No train normal samples available")
    args.support_paths = {sample[0] for sample in samples}
    return DataLoader(
        RFPathDataset(samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def build_eval_loaders(args):
    loaders = []
    for signal in args.signals:
        for scene in SCENES:
            for jsr in JSR_BY_SIGNAL[signal]:
                samples = collect_samples(
                    signal,
                    scene,
                    jsr,
                    "test",
                    k_shot=args.k_shot,
                    exclude_paths=getattr(args, "support_paths", set()),
                )
                loader = DataLoader(
                    RFPathDataset(samples),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                loaders.append((signal, scene, jsr, loader))
    return loaders


def to_model_input(model, raw_batch, device, rgb_from_bgr=True):
    # Target-scene OFDMA samples are already 240x240 uint8 tensors after the
    # official letterbox preprocessing.  Avoid a per-image PIL/torchvision
    # round trip for this fixed geometry; this is numerically equivalent to
    # ToTensor + CLIP normalization and keeps the formal protocol unchanged.
    if (
        isinstance(raw_batch, torch.Tensor)
        and raw_batch.ndim == 4
        and raw_batch.shape[-1] == 3
        and getattr(model, "dataset_name", None) == "ofdma_spectrum"
        and int(raw_batch.shape[1]) == int(getattr(model, "out_size_h", -1))
        and int(raw_batch.shape[2]) == int(getattr(model, "out_size_w", -1))
    ):
        batch = raw_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        if rgb_from_bgr:
            batch = batch[..., [2, 1, 0]]
        batch = batch.permute(0, 3, 1, 2).div_(255.0)
        mean = batch.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = batch.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        return (batch - mean) / std

    imgs = []
    for arr in raw_batch:
        arr = arr.numpy()
        if rgb_from_bgr:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        imgs.append(model.transform(Image.fromarray(arr)))
    return torch.stack(imgs, dim=0).to(device)


def save_checkpoint(model, path):
    state_dict = model.state_dict()
    selected = {}
    for k, v in state_dict.items():
        if (
            k in {"feature_gallery1", "feature_gallery2", "text_features"}
            or k.startswith("prompt_learner.")
            or k.startswith("visual_adapters.")
            or k.startswith("visual_class_prompt_adapter.")
            or ".lora_" in k
        ):
            selected[k] = v
    torch.save(selected, path)


@torch.no_grad()
def build_gallery(model, train_loader, device):
    model.eval_mode()
    global_features, features1, features2 = [], [], []
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Build pooled gallery", leave=False):
        data = to_model_input(model, data, device, rgb_from_bgr=True)
        cls_feature, _, feature_map1, feature_map2 = model.encode_image(data)
        global_features.append(cls_feature)
        features1.append(feature_map1)
        features2.append(feature_map2)
    global_features = torch.cat(global_features, dim=0)
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2, global_features)
    model.set_visual_class_prototype(global_features)


def train_one_epoch_cls(model, train_loader, args, device, criterion, criterion_tip):
    model.train()
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Train cls", leave=False):
        data = to_model_input(model, data, device, rgb_from_bgr=True)
        normal_prompt, abnormal_handle, abnormal_learned = model.prompt_learner()
        normal_features = model.encode_text_embedding(normal_prompt, model.tokenized_normal_prompts)
        abnormal_handle_features = model.encode_text_embedding(abnormal_handle, model.tokenized_abnormal_prompts_handle)
        abnormal_learned_features = model.encode_text_embedding(abnormal_learned, model.tokenized_abnormal_prompts_learned)
        abnormal_features = torch.cat([abnormal_handle_features, abnormal_learned_features], dim=0)

        mean_handle = torch.mean(F.normalize(abnormal_handle_features, dim=-1), dim=0)
        mean_learned = torch.mean(F.normalize(abnormal_learned_features, dim=-1), dim=0)
        loss_match = (mean_handle - mean_learned).norm(dim=0) ** 2.0

        visual_features = model.encode_image(data)
        cls_feature = visual_features[0]
        normal_anchor = F.normalize(normal_features.mean(dim=0).unsqueeze(0), dim=-1)
        abnormal_anchor = F.normalize(abnormal_features.mean(dim=0).unsqueeze(0), dim=-1)
        abnormal_features = F.normalize(abnormal_features, dim=-1)

        l_pos = torch.einsum("nc,cm->nm", cls_feature, normal_anchor.transpose(0, 1))
        l_neg = torch.einsum("nc,cm->nm", cls_feature, abnormal_features.transpose(0, 1))
        logit_scale = model.model.logit_scale.half() if model.precision == "fp16" else model.model.logit_scalef
        logits = torch.cat([l_pos, l_neg], dim=-1) * logit_scale
        target = torch.zeros([logits.shape[0]], dtype=torch.long).to(device)
        loss = criterion(logits, target) + criterion_tip(cls_feature, normal_anchor, abnormal_anchor)
        loss = loss + loss_match * args.lambda1

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()


def train_one_epoch_seg(model, train_loader, args, device, criterion, criterion_tip):
    model.train()
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Train seg", leave=False):
        data = to_model_input(model, data, device, rgb_from_bgr=True)
        normal_prompt, abnormal_handle, abnormal_learned = model.prompt_learner()
        normal_features = model.encode_text_embedding(normal_prompt, model.tokenized_normal_prompts)
        abnormal_handle_features = model.encode_text_embedding(abnormal_handle, model.tokenized_abnormal_prompts_handle)
        abnormal_learned_features = model.encode_text_embedding(abnormal_learned, model.tokenized_abnormal_prompts_learned)
        abnormal_features = torch.cat([abnormal_handle_features, abnormal_learned_features], dim=0)

        mean_handle = torch.mean(F.normalize(abnormal_handle_features, dim=-1), dim=0)
        mean_learned = torch.mean(F.normalize(abnormal_learned_features, dim=-1), dim=0)
        loss_match = (mean_handle - mean_learned).norm(dim=0) ** 2.0

        visual_features = model.encode_image(data)
        feature_map = visual_features[1]
        normal_anchor = F.normalize(normal_features.mean(dim=0).unsqueeze(0), dim=-1)
        abnormal_anchor = F.normalize(abnormal_features.mean(dim=0).unsqueeze(0), dim=-1)
        abnormal_features = F.normalize(abnormal_features, dim=-1)

        l_pos = torch.einsum("nic,cj->nij", feature_map, normal_anchor.transpose(0, 1))
        l_neg = torch.einsum("nic,cj->nij", feature_map, abnormal_features.transpose(0, 1))
        logit_scale = model.model.logit_scale.half() if model.precision == "fp16" else model.model.logit_scalef
        logits = torch.cat([l_pos, l_neg], dim=-1) * logit_scale
        target = torch.zeros([logits.shape[0], logits.shape[1]], dtype=torch.long).to(device)
        loss = criterion(logits.transpose(1, 2), target)
        loss = loss + criterion_tip(feature_map, normal_anchor, abnormal_anchor)
        loss = loss + loss_match * args.lambda1

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()


@torch.no_grad()
def evaluate_cls(model, eval_loaders, args, device):
    model.eval_mode()
    model.build_text_feature_gallery()
    rows = []
    for signal, scene, jsr, loader in eval_loaders:
        scores_img, gt_list, score_maps = [], [], []
        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval cls {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            vf = model.encode_image(data_t)
            score_img, score_map = model.score_cached(vf, "cls")
            scores_img.extend(score_img)
            score_maps.extend(score_map)
            gt_list.extend(label.numpy().tolist())
        metrics = metric_cal_img(np.asarray(scores_img), gt_list, np.asarray(score_maps))
        rows.append({"dataset": signal, "scene": scene, "jsr": jsr, "i_roc": metrics["i_roc"]})
    return rows


@torch.no_grad()
def evaluate_seg(model, eval_loaders, args, device):
    model.eval_mode()
    model.build_text_feature_gallery()
    rows = []
    for signal, scene, jsr, loader in eval_loaders:
        score_maps, gt_mask_list, test_imgs = [], [], []
        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval seg {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            for d, m in zip(data_t, mask):
                test_imgs.append(np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8))
                m = m.numpy()
                m[m > 0] = 1
                gt_mask_list.append(m)
            maps = model(data_t, "seg")
            score_maps.extend(maps)
        test_imgs, score_maps, gt_mask_list = specify_resolution(
            test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution)
        )
        metrics = metric_cal_pix(np.asarray(score_maps), gt_mask_list)
        rows.append({"dataset": signal, "scene": scene, "jsr": jsr, "p_roc": metrics["p_roc"]})
    return rows


def macro_score(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def write_rows(path, rows, task, method, checkpoint, metric_key):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "task", "dataset", "scene", "jsr", metric_key, "checkpoint"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": method,
                "task": task,
                "dataset": row["dataset"],
                "scene": row["scene"],
                "jsr": row["jsr"],
                metric_key: f"{float(row[metric_key]):.4f}",
                "checkpoint": str(checkpoint),
            })


def apply_method_preset(args):
    for k, v in METHOD_PRESETS.get(args.method, {}).items():
        setattr(args, k, v)


def write_run_metadata(path, args, model, prompt_dataset, prompt_class):
    prompt_learner = getattr(model, "prompt_learner", None)
    grouped_slices = getattr(prompt_learner, "abnormal_handle_group_slices", []) if prompt_learner is not None else []
    metadata = {
        "method": args.method,
        "task": args.task,
        "seed": args.seed,
        "epochs": args.epochs,
        "input_mode": args.input_mode,
        "prompt_dataset": prompt_dataset,
        "prompt_class": prompt_class,
        "is_rf_prompt_class": is_rf_prompt_class(prompt_class, prompt_dataset),
        "rf_signal_key": get_rf_signal_key(prompt_class, prompt_dataset),
        "prompt_mode": args.prompt_mode,
        "text_prototype_mode": args.text_prototype_mode,
        "visual_class_prompt": args.visual_class_prompt,
        "visual_adapter": args.visual_adapter,
        "grouped_prompt_specs": [spec[0] for spec in get_grouped_rf_abnormal_prompt_specs()],
        "grouped_slice_count": len(grouped_slices),
        "text_feature_count": int(getattr(model, "text_features", torch.empty(0)).shape[0]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("[run_metadata] " + json.dumps(metadata, ensure_ascii=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cls", "seg"], required=True)
    parser.add_argument("--method", choices=sorted(METHOD_PRESETS), required=True)
    parser.add_argument("--output-root", default="analysis_outputs/20260627_method_funnel")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--normal-sampling", choices=["all", "frequency_one_per_band", "split_75_25"], default="frequency_one_per_band")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", default="laion400m_e32")
    parser.add_argument("--prompt-mode", default="rf")
    parser.add_argument("--input-mode", default="rgb")
    parser.add_argument("--text-prototype-mode", default="single")
    parser.add_argument("--cls-score-mode", default="text_only")
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--visual-class-prompt", action="store_true", default=False)
    parser.add_argument("--visual-class-token-num", type=int, default=2)
    parser.add_argument("--visual-class-prompt-bottleneck-ratio", type=float, default=0.25)
    parser.add_argument("--visual-class-prompt-alpha", type=float, default=0.2)
    parser.add_argument("--visual-class-prototype-mode", default="mean")
    parser.add_argument("--visual-class-prototype-num", type=int, default=1)
    parser.add_argument("--visual-adapter", action="store_true", default=False)
    parser.add_argument("--adapter-bottleneck-ratio", type=float, default=0.25)
    parser.add_argument("--adapter-alpha", type=float, default=0.2)
    parser.add_argument("--visual-lora", action="store_true", default=False)
    parser.add_argument("--learnable-score-fusion", action="store_true", default=False)
    parser.add_argument("--text-aligned-dense", action="store_true", default=False)
    parser.add_argument("--use-cpu", type=int, default=0)
    args = parser.parse_args()

    apply_method_preset(args)
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu else "cuda:0"

    train_loader = build_train_loader(args)
    eval_loaders = build_eval_loaders(args)
    prompt_dataset = "rf_target_test_pool"
    prompt_class = "signal"
    kwargs = vars(args).copy()
    kwargs.update({
        "dataset": prompt_dataset,
        "class_name": prompt_class,
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    build_gallery(model, train_loader, device)

    parameter_groups = model.trainable_parameter_groups(
        base_lr=args.lr,
        prompt_lr=None,
        visual_adapter_lr=None,
        visual_class_prompt_lr=None,
        visual_lora_lr=None,
    )
    args.optimizer = torch.optim.SGD(
        parameter_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    out_root = Path(args.output_root)
    run_root = out_root / "runs" / args.method / args.task
    checkpoint = run_root / "checkpoint" / "overall-best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_root / "run_metadata.json", args, model, prompt_dataset, prompt_class)
    best_score = None
    best_rows = None
    metric_key = "i_roc" if args.task == "cls" else "p_roc"

    for epoch in range(args.epochs):
        if args.task == "cls":
            train_one_epoch_cls(model, train_loader, args, device, criterion, criterion_tip)
        else:
            train_one_epoch_seg(model, train_loader, args, device, criterion, criterion_tip)
        scheduler.step()
        if model.use_visual_adapter or model.use_visual_lora:
            build_gallery(model, train_loader, device)

        should_eval = args.task == "cls" or ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
        if not should_eval:
            continue
        rows = evaluate_cls(model, eval_loaders, args, device) if args.task == "cls" else evaluate_seg(model, eval_loaders, args, device)
        score = macro_score(rows, metric_key)
        print(f"Epoch {epoch + 1}/{args.epochs} {metric_key} macro={score:.4f}")
        if best_score is None or score > best_score:
            best_score = score
            best_rows = rows
            save_checkpoint(model, checkpoint)

    result_path = out_root / f"results_{args.task}_{args.method}.csv"
    write_rows(result_path, best_rows or [], args.task, args.method, checkpoint, metric_key)
    print(f"best {metric_key} macro={best_score:.4f}")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
