"""RF cross-library dense mask supervision entrypoint.

Formal protocol:
1. train the dense anomaly-map head with public source normal/abnormal/mask;
2. build the target-domain normal gallery from target train normal samples only;
3. evaluate on the project RF target test split.

The entrypoint deliberately has no source-only cross-library mode.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import denormalization, get_dataloader_from_args
from PromptAD import PromptAD
from train_cls import (
    get_args as get_base_args,
    save_image_scores,
    str2bool,
)
from utils.eval_utils import specify_resolution
from utils.metrics import metric_cal_img
from utils.training_utils import setup_seed


TASK = 'RF_PUBLIC_TO_TARGET_DENSE_ADAPT'
SOURCE_DATASET = 'rf_public_pooled_smoke'
SOURCE_CLASS = 'signal'
TARGET_DATASET = 'rf_target_test_pool'
TARGET_CLASSES = ['burst', 'chirp', 'dsss']

DEFAULT_ROOT = 'analysis_outputs/promptad_rf_cross_public_dense_adapt'
DEFAULT_RESULTS_CSV = f'{DEFAULT_ROOT}/results.csv'

CSV_HEADER = [
    'method', 'seed', 'class', 'jsr', 'best_epoch',
    'iroc', 'sp_ap', 'px_auroc', 'px_ap',
    'checkpoint', 'notes',
]

# 多 JSR 评估的目标档位; 'avg' 表示宏平均(跨档等权)
EVAL_JSRS = ['m10db', 'm20db', 'm30db']


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-noise-level', type=str, default='m10db')
    parser.add_argument('--target-noise-level', type=str, default='m10db',
                        help='legacy 单档评估; 若 --eval-jsrs 给了多档则忽略此项')
    parser.add_argument('--eval-jsrs', type=str, default='m10db,m20db,m30db',
                        help='逗号分隔, 每个 epoch 对 target 在这些 JSR 上分别评估; '
                             "best 按各类'跨档平均 i_roc'选; 设为单档(如 'm10db')即单档评估")
    parser.add_argument('--cross-root-dir', type=str, default=DEFAULT_ROOT)
    parser.add_argument('--method-tag', type=str, default='dense_A')
    parser.add_argument('--results-csv', type=str, default=DEFAULT_RESULTS_CSV)
    parser.add_argument('--max-normal-per-class', type=int, default=300)
    parser.add_argument('--max-abnormal-per-class', type=int, default=300)
    parser.add_argument('--pool-classes', type=str, default='burst,chirp,dsss')
    parser.add_argument('--target-classes', type=str, default='burst,chirp,dsss')
    parser.add_argument('--dense-lr', type=float, default=1e-3)
    parser.add_argument('--dense-bce-weight', type=float, default=1.0)
    parser.add_argument('--dense-dice-weight', type=float, default=1.0)
    parser.add_argument('--dense-score-alpha', type=float, default=0.05)
    parser.add_argument('--dense-eval-mode', type=str, default='dense_fusion',
                        choices=['dense_fusion', 'dense_only', 'dense_map_only', 'text_only'],
                        help='dense 分支评估模式: dense_map_only 表示 image 分数用原 PromptAD, pixel map 用 dense')
    parser.add_argument('--image-roc-source', type=str, default='harmonic', choices=['harmonic', 'raw'],
                        help='harmonic 使用旧 metric_cal_img(img,map); raw 直接用 image score 算 Image ROC')
    parser.add_argument('--freeze-prompt', type=str2bool, choices=[True, False], default=True)
    parser.add_argument('--eval-only', type=str2bool, choices=[True, False], default=False,
                        help='只加载 dense checkpoint 做 target normal gallery 评估, 不重新训练')
    parser.add_argument('--dense-checkpoint', type=str, default='',
                        help='--eval-only 时加载的 dense/prompt checkpoint 路径')
    parser.add_argument('--results-notes', type=str, default='')
    dense_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_base_args()
    finally:
        sys.argv = original_argv

    args.source_noise_level = dense_args.source_noise_level
    args.target_noise_level = dense_args.target_noise_level
    args.eval_jsrs = [j.strip() for j in dense_args.eval_jsrs.split(',') if j.strip()]
    if not args.eval_jsrs:
        args.eval_jsrs = [args.target_noise_level]
    args.cross_root_dir = dense_args.cross_root_dir
    args.method_tag = dense_args.method_tag
    args.results_csv = dense_args.results_csv
    args.max_normal_per_class = dense_args.max_normal_per_class
    args.max_abnormal_per_class = dense_args.max_abnormal_per_class
    args.pool_classes = [c.strip() for c in dense_args.pool_classes.split(',') if c.strip()]
    args.target_classes = [c.strip() for c in dense_args.target_classes.split(',') if c.strip()]
    invalid = [c for c in args.target_classes if c not in TARGET_CLASSES]
    if invalid:
        raise ValueError(f'--target-classes has invalid values {invalid}; allowed: {TARGET_CLASSES}')
    args.dense_lr = dense_args.dense_lr
    args.dense_bce_weight = dense_args.dense_bce_weight
    args.dense_dice_weight = dense_args.dense_dice_weight
    args.dense_score_alpha = dense_args.dense_score_alpha
    args.dense_eval_mode = dense_args.dense_eval_mode
    args.image_roc_source = dense_args.image_roc_source
    args.freeze_prompt = dense_args.freeze_prompt
    args.eval_only = dense_args.eval_only
    args.dense_checkpoint = dense_args.dense_checkpoint
    args.results_notes = dense_args.results_notes
    args.split_mode = 'normal_75_25'
    args.dense_mask_branch = True
    args.dense_mask_score_alpha = dense_args.dense_score_alpha
    return args


def make_loader_args(args, dataset, class_name, noise_level, **extra):
    kwargs = vars(args).copy()
    kwargs['dataset'] = dataset
    kwargs['class_name'] = class_name
    kwargs['noise_level'] = noise_level
    kwargs.update(extra)
    return kwargs


def get_run_dirs(args):
    pool_tag = '_'.join(args.pool_classes) if len(args.pool_classes) < len(TARGET_CLASSES) else 'pooled'
    mode = f'{args.input_mode}_{args.prompt_mode}_dense_targetnorm'
    root = os.path.join(
        args.cross_root_dir,
        'runs',
        f'method_{args.method_tag}',
        pool_tag,
        mode,
        f'seed_{args.seed}',
    )
    check_dir = os.path.join(root, 'checkpoint')
    score_dir = os.path.join(root, 'scores')
    for path in (check_dir, score_dir):
        os.makedirs(path, exist_ok=True)
    check_paths = {
        cls: os.path.join(check_dir, f'{TASK}-method_{args.method_tag}-{pool_tag}_train-{cls}_test-seed{args.seed}-best.pt')
        for cls in args.target_classes
    }
    check_paths['all'] = os.path.join(
        check_dir,
        f'{TASK}-method_{args.method_tag}-{pool_tag}_train-all_test-seed{args.seed}-overall-best.pt',
    )
    # score 按 (class, jsr) 分别落盘, 便于事后做合并 micro ROC
    score_paths = {
        (cls, jsr): os.path.join(
            score_dir,
            f'method_{args.method_tag}-{pool_tag}_train-{cls}_{jsr}_test-seed{args.seed}-image_scores.npz',
        )
        for cls in args.target_classes
        for jsr in args.eval_jsrs
    }
    return check_paths, score_paths


def append_result(results_csv, row):
    os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
    new_file = not os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def to_model_input(model, raw_batch, device):
    """Accept either:
       - tensor batch (worker 已做完 transform): shape [B,C,H,W] -> 直接 .to(device)
       - raw uint8 ndarray batch (legacy 主进程 transform 路径)
    """
    if torch.is_tensor(raw_batch):
        # DataLoader 默认 collate 会把同一类型 stack 成 [B,...] 的 tensor
        return raw_batch.to(device, non_blocking=True)
    imgs = []
    for arr in raw_batch:
        arr = arr.numpy()
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        imgs.append(model.transform(Image.fromarray(arr)))
    return torch.stack(imgs, dim=0).to(device)


def downsample_mask(mask, size, device):
    mask = mask.float().to(device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = (mask > 0).float()
    return F.adaptive_max_pool2d(mask, output_size=size)


def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def dense_mask_loss(raw_logits, post_logits, target, bce_weight, dice_weight):
    bce = F.binary_cross_entropy_with_logits(raw_logits, target)
    bce = bce + F.binary_cross_entropy_with_logits(post_logits, target)
    dice = dice_loss_from_logits(raw_logits, target) + dice_loss_from_logits(post_logits, target)
    return bce_weight * bce + dice_weight * dice


def save_dense_checkpoint(model, path):
    state = model.state_dict()
    keep = {
        k: v for k, v in state.items()
        if k.startswith('dense_mask_head.')
        or k.startswith('prompt_learner.')
    }
    torch.save(keep, path)


@torch.no_grad()
def build_normal_gallery(model, loader, device, desc='Target normal gallery'):
    model.eval_mode()
    features1, features2, global_features = [], [], []
    for data, mask, label, name, img_type in tqdm(loader, desc=desc, leave=False):
        data = to_model_input(model, data, device)
        cls_feature, _, feature_map1, feature_map2 = model.encode_image(data)
        global_features.append(cls_feature)
        features1.append(feature_map1)
        features2.append(feature_map2)
    if not global_features:
        raise RuntimeError(f'{desc} is empty; target train normal adaptation cannot be built.')
    model.build_image_feature_gallery(torch.cat(features1, dim=0), torch.cat(features2, dim=0), torch.cat(global_features, dim=0))
    model.set_visual_class_prototype(torch.cat(global_features, dim=0))


def train_one_epoch(model, loader, optimizer, args, device, epoch_idx):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Dense source train {epoch_idx}/{args.Epoch}', leave=False)
    for step, (data, mask, label, name, img_type) in enumerate(pbar, start=1):
        data = to_model_input(model, data, device)
        target = downsample_mask(mask, model.grid_size, device)
        visual_features = model.encode_image(data)
        raw_logits, post_logits = model.calculate_dense_mask_logits(visual_features)
        loss = dense_mask_loss(
            raw_logits,
            post_logits,
            target,
            args.dense_bce_weight,
            args.dense_dice_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        pbar.set_postfix(loss=f'{loss.item():.4f}', avg=f'{total_loss / step:.4f}')
    return total_loss / max(1, step)


@torch.no_grad()
def evaluate_target(model, loader, args, device, label_tag):
    model.eval_mode()
    scores_img, score_maps = [], []
    visual_maps_raw, text_scores_collected = [], []
    test_imgs, gt_list, gt_mask_list, names = [], [], [], []

    for raw_data, mask, label, name, img_type in tqdm(loader, desc=f'Eval {label_tag}', leave=False):
        data = to_model_input(model, raw_data, device)
        visual_features = model.encode_image(data)
        score_img, score_map, raw_maps, text_scores = model.score_cached(
            visual_features,
            'cls',
            return_raw_map=True,
            image=data,
        )
        test_imgs += [denormalization(d.cpu().numpy()) for d in data]
        names += list(name)
        gt_list += label.numpy().tolist()
        for m in mask.numpy():
            m[m > 0] = 1
            gt_mask_list.append(m)
        scores_img += score_img
        score_maps += score_map
        visual_maps_raw += raw_maps
        text_scores_collected += list(text_scores)

    test_imgs, score_maps, gt_mask_list = specify_resolution(
        test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
    return scores_img, score_maps, gt_list, gt_mask_list, names, visual_maps_raw, text_scores_collected


def compute_metrics(scores_img, score_maps, gt_list, gt_mask_list, image_roc_source='harmonic'):
    img_scores = np.asarray(scores_img, dtype=np.float64)
    map_scores = np.asarray(score_maps, dtype=np.float64)
    gt = np.asarray(gt_list, dtype=int)

    max_map_scores = map_scores.reshape(map_scores.shape[0], -1).max(axis=1)
    safe_img = np.where(img_scores == 0, 1e-12, img_scores)
    safe_map = np.where(max_map_scores == 0, 1e-12, max_map_scores)
    fused = 1.0 / (1.0 / safe_map + 1.0 / safe_img)

    if image_roc_source == 'raw':
        i_roc = roc_auc_score(gt, img_scores) * 100
        sp_ap = average_precision_score(gt, img_scores) * 100
    else:
        img_dict = metric_cal_img(img_scores, gt, map_scores)
        i_roc = float(img_dict['i_roc'])
        sp_ap = average_precision_score(gt, fused) * 100

    gt_mask = np.asarray(gt_mask_list, dtype=int)
    px_gt = gt_mask.flatten()
    px_score = map_scores.flatten()
    if px_gt.sum() == 0:
        px_auroc = float('nan')
        px_ap = float('nan')
    else:
        px_auroc = roc_auc_score(px_gt, px_score) * 100
        px_ap = average_precision_score(px_gt, px_score) * 100
    return {
        'i_roc': float(i_roc),
        'sp_ap': float(sp_ap),
        'px_auroc': float(px_auroc),
        'px_ap': float(px_ap),
    }


def _mean_safe(vals):
    """跨档求宏平均, 忽略 NaN; 全 NaN 返回 NaN."""
    arr = np.asarray([v for v in vals if not (v is None or np.isnan(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size > 0 else float('nan')


def fit(model, args, source_query_loader, target_adapt_loaders, target_test_loaders, device, check_paths, score_paths):
    """每 epoch 对每个 (class, jsr) 评估.

    - best[cls]: 按该 class 跨 JSR 平均 i_roc 挑 best epoch;
    - best_overall: 按全部 class × JSR 的宏平均 i_roc 挑 best epoch.

    best[cls] / best_overall 结构:
        {
            'epoch': int,
            'avg_iroc': float,                          # 用来挑 best 的标量
            'per_jsr': {jsr: metrics_dict, ...},        # class best 时为该类三档指标
            'per_class': {cls: {jsr: metrics_dict}},    # overall best 时为全量指标
        }
    """
    if model.dense_mask_head is None:
        raise RuntimeError('dense_mask_head was not created')
    if args.freeze_prompt:
        for p in model.prompt_learner.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.dense_mask_head.parameters(), lr=args.dense_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=max(args.dense_lr * 0.01, 1e-6))

    best = {cls: {'avg_iroc': -1.0, 'epoch': -1, 'per_jsr': None} for cls in args.target_classes}
    best_overall = {'avg_iroc': -1.0, 'epoch': -1, 'per_class': None}
    model.build_text_feature_gallery()

    for epoch in range(1, args.Epoch + 1):
        loss = train_one_epoch(model, source_query_loader, optimizer, args, device, epoch)
        scheduler.step()
        model.build_text_feature_gallery()

        msg_parts = []
        epoch_per_class = {}
        for cls in args.target_classes:
            per_jsr_metrics = {}
            per_jsr_payload = {}
            for jsr in args.eval_jsrs:
                build_normal_gallery(
                    model,
                    target_adapt_loaders[(cls, jsr)],
                    device,
                    desc=f'Target normal gallery {cls}@{jsr}',
                )
                scores_img, score_maps, gt_list, gt_mask_list, names, raw_maps, text_scores = evaluate_target(
                    model, target_test_loaders[(cls, jsr)], args, device, f'{cls}@{jsr}@ep{epoch}')
                cur = compute_metrics(scores_img, score_maps, gt_list, gt_mask_list, args.image_roc_source)
                per_jsr_metrics[jsr] = cur
                per_jsr_payload[jsr] = (names, scores_img, gt_list, raw_maps, text_scores)

            epoch_per_class[cls] = per_jsr_metrics
            avg_iroc = _mean_safe([per_jsr_metrics[j]['i_roc'] for j in args.eval_jsrs])
            avg_px = _mean_safe([per_jsr_metrics[j]['px_auroc'] for j in args.eval_jsrs])

            if avg_iroc > best[cls]['avg_iroc']:
                best[cls] = {
                    'avg_iroc': avg_iroc,
                    'avg_px_auroc': avg_px,
                    'epoch': epoch,
                    'per_jsr': per_jsr_metrics,
                }
                save_dense_checkpoint(model, check_paths[cls])
                for jsr, payload in per_jsr_payload.items():
                    names, scores_img, gt_list, raw_maps, text_scores = payload
                    save_image_scores(names, scores_img, gt_list, score_paths[(cls, jsr)],
                                      visual_maps=raw_maps, text_scores=text_scores)

            jsr_str = ' '.join(f"{j}:{per_jsr_metrics[j]['i_roc']:.1f}/{per_jsr_metrics[j]['px_auroc']:.1f}"
                               for j in args.eval_jsrs)
            msg_parts.append(
                f"{cls} [{jsr_str}] avg={avg_iroc:.2f} best_avg={best[cls]['avg_iroc']:.2f}@{best[cls]['epoch']}"
            )

        overall_avg_iroc = _mean_safe(
            [epoch_per_class[cls][jsr]['i_roc'] for cls in args.target_classes for jsr in args.eval_jsrs]
        )
        overall_avg_px = _mean_safe(
            [epoch_per_class[cls][jsr]['px_auroc'] for cls in args.target_classes for jsr in args.eval_jsrs]
        )
        if overall_avg_iroc > best_overall['avg_iroc']:
            best_overall = {
                'avg_iroc': overall_avg_iroc,
                'avg_px_auroc': overall_avg_px,
                'epoch': epoch,
                'per_class': epoch_per_class,
            }
            save_dense_checkpoint(model, check_paths['all'])

        print(
            f"Epoch [{epoch}/{args.Epoch}] loss={loss:.4f} | "
            + ' | '.join(msg_parts)
            + f" | all_avg={overall_avg_iroc:.2f} best_all={best_overall['avg_iroc']:.2f}@{best_overall['epoch']}"
        )
    return best, best_overall


@torch.no_grad()
def evaluate_once(model, args, target_adapt_loaders, target_test_loaders, device, score_paths):
    model.eval_mode()
    model.build_text_feature_gallery()
    best = {}
    overall_per_class = {}

    for cls in args.target_classes:
        per_jsr_metrics = {}
        for jsr in args.eval_jsrs:
            build_normal_gallery(
                model,
                target_adapt_loaders[(cls, jsr)],
                device,
                desc=f'Target normal gallery {cls}@{jsr}',
            )
            scores_img, score_maps, gt_list, gt_mask_list, names, raw_maps, text_scores = evaluate_target(
                model, target_test_loaders[(cls, jsr)], args, device, f'{cls}@{jsr}@loaded')
            cur = compute_metrics(scores_img, score_maps, gt_list, gt_mask_list, args.image_roc_source)
            per_jsr_metrics[jsr] = cur
            save_image_scores(names, scores_img, gt_list, score_paths[(cls, jsr)],
                              visual_maps=raw_maps, text_scores=text_scores)

        overall_per_class[cls] = per_jsr_metrics
        best[cls] = {
            'avg_iroc': _mean_safe([per_jsr_metrics[j]['i_roc'] for j in args.eval_jsrs]),
            'avg_px_auroc': _mean_safe([per_jsr_metrics[j]['px_auroc'] for j in args.eval_jsrs]),
            'epoch': 'loaded',
            'per_jsr': per_jsr_metrics,
        }

    best_overall = {
        'avg_iroc': _mean_safe([overall_per_class[cls][jsr]['i_roc'] for cls in args.target_classes for jsr in args.eval_jsrs]),
        'avg_px_auroc': _mean_safe([overall_per_class[cls][jsr]['px_auroc'] for cls in args.target_classes for jsr in args.eval_jsrs]),
        'epoch': 'loaded',
        'per_class': overall_per_class,
    }
    return best, best_overall


def main(args):
    if args.seed is None:
        args.seed = 111
    setup_seed(args.seed)

    device = 'cuda:0' if args.use_cpu == 0 else 'cpu'
    args.device = device
    args.out_size_h = args.resolution
    args.out_size_w = args.resolution
    args.cls_score_mode = args.dense_eval_mode

    source_kwargs = make_loader_args(
        args, SOURCE_DATASET, SOURCE_CLASS, args.source_noise_level,
        max_normal_per_class=args.max_normal_per_class,
        max_abnormal_per_class=args.max_abnormal_per_class,
        pool_classes=args.pool_classes,
    )

    # 先建 model, 这样 dataset 一开始就拿到 model.transform, worker 直接做预处理.
    model_kwargs = vars(args).copy()
    model_kwargs['dataset'] = SOURCE_DATASET
    model_kwargs['class_name'] = SOURCE_CLASS
    model_kwargs['noise_level'] = args.source_noise_level
    model_kwargs['dense_mask_branch'] = True
    model_kwargs['dense_mask_score_alpha'] = args.dense_score_alpha
    model = PromptAD(**model_kwargs).to(device)
    if args.eval_only:
        if not args.dense_checkpoint:
            raise ValueError('--eval-only requires --dense-checkpoint')
        state = torch.load(args.dense_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[eval-only] loaded dense checkpoint: {args.dense_checkpoint}')
        if unexpected:
            print(f'[eval-only] unexpected checkpoint keys: {unexpected}')

    def _attach(loader):
        if hasattr(loader.dataset, 'set_sample_transform'):
            loader.dataset.set_sample_transform(model.transform)
        return loader

    source_query_loader, _ = get_dataloader_from_args(phase='test', perturbed=False, **source_kwargs)
    _attach(source_query_loader)

    # 多档 target loader: train normal 用于 adaptation, test 用于最终评估.
    # key 为 (class, jsr), 避免用 m10db 的正常库去评估 m20db/m30db.
    target_adapt_loaders = {}
    target_test_loaders = {}
    for cls in args.target_classes:
        for jsr in args.eval_jsrs:
            target_kwargs = make_loader_args(args, TARGET_DATASET, cls, jsr)
            target_adapt_loaders[(cls, jsr)], _ = get_dataloader_from_args(phase='train', perturbed=False, **target_kwargs)
            target_test_loaders[(cls, jsr)], _ = get_dataloader_from_args(phase='test', perturbed=False, **target_kwargs)
            _attach(target_adapt_loaders[(cls, jsr)])
            _attach(target_test_loaders[(cls, jsr)])
    print(f'[multi-JSR] eval grid = {len(args.target_classes)} classes × {len(args.eval_jsrs)} jsrs '
          f'= {len(target_test_loaders)} test loaders + {len(target_adapt_loaders)} target-normal loaders '
          f'({", ".join(args.eval_jsrs)})')
    print('[multi-JSR] injected model.transform into datasets (worker-side preprocessing)')

    check_paths, score_paths = get_run_dirs(args)
    if args.eval_only and args.dense_checkpoint:
        for key in list(check_paths.keys()):
            check_paths[key] = args.dense_checkpoint

    started = time.time()
    if args.eval_only:
        best, best_overall = evaluate_once(
            model, args, target_adapt_loaders, target_test_loaders, device, score_paths
        )
    else:
        best, best_overall = fit(
            model, args, source_query_loader, target_adapt_loaders, target_test_loaders, device, check_paths, score_paths
        )
    elapsed = time.time() - started

    notes = args.results_notes or (
        f'dense_mask_source_supervision+target_train_normal_gallery+dense_fusion;'
        f'epoch={"loaded" if args.eval_only else args.Epoch};mode={args.dense_eval_mode};image_roc={args.image_roc_source};alpha={args.dense_score_alpha};elapsed={elapsed:.0f}s'
    )

    def _fmt(v):
        return f'{v:.4f}' if not (v is None or np.isnan(v)) else 'NA'

    # 长表写入: 每 class × 每 jsr 一行 + 每 class 一行 avg + all avg.
    for cls in args.target_classes:
        rec = best[cls]
        per_jsr = rec['per_jsr']
        for jsr in args.eval_jsrs:
            m = per_jsr[jsr]
            append_result(args.results_csv, {
                'method': args.method_tag,
                'seed': args.seed,
                'class': cls,
                'jsr': jsr,
                'best_epoch': rec['epoch'],
                'iroc': _fmt(m['i_roc']),
                'sp_ap': _fmt(m['sp_ap']),
                'px_auroc': _fmt(m['px_auroc']),
                'px_ap': _fmt(m['px_ap']),
                'checkpoint': check_paths[cls],
                'notes': notes,
            })
        # 平均行: avg 是按 best epoch 的三档值算的宏平均
        avg_sp_ap = _mean_safe([per_jsr[j]['sp_ap'] for j in args.eval_jsrs])
        avg_px_ap = _mean_safe([per_jsr[j]['px_ap'] for j in args.eval_jsrs])
        append_result(args.results_csv, {
            'method': args.method_tag,
            'seed': args.seed,
            'class': cls,
            'jsr': 'avg',
            'best_epoch': rec['epoch'],
            'iroc': _fmt(rec['avg_iroc']),
            'sp_ap': _fmt(avg_sp_ap),
            'px_auroc': _fmt(rec['avg_px_auroc']),
            'px_ap': _fmt(avg_px_ap),
            'checkpoint': check_paths[cls],
            'notes': notes,
        })
        print(f"[DONE] {cls}: best_ep={rec['epoch']} avg_iroc={rec['avg_iroc']:.2f} "
              f"avg_px={rec['avg_px_auroc']:.2f} | "
              + ' '.join(f"{j}={per_jsr[j]['i_roc']:.1f}/{per_jsr[j]['px_auroc']:.1f}" for j in args.eval_jsrs))
    all_per_class = best_overall['per_class']
    all_sp_ap = _mean_safe(
        [all_per_class[cls][jsr]['sp_ap'] for cls in args.target_classes for jsr in args.eval_jsrs]
    )
    all_px_ap = _mean_safe(
        [all_per_class[cls][jsr]['px_ap'] for cls in args.target_classes for jsr in args.eval_jsrs]
    )
    append_result(args.results_csv, {
        'method': args.method_tag,
        'seed': args.seed,
        'class': 'all',
        'jsr': 'avg',
        'best_epoch': best_overall['epoch'],
        'iroc': _fmt(best_overall['avg_iroc']),
        'sp_ap': _fmt(all_sp_ap),
        'px_auroc': _fmt(best_overall['avg_px_auroc']),
        'px_ap': _fmt(all_px_ap),
        'checkpoint': check_paths['all'],
        'notes': notes,
    })
    print(f"[DONE] all: best_ep={best_overall['epoch']} avg_iroc={best_overall['avg_iroc']:.2f} "
          f"avg_px={best_overall['avg_px_auroc']:.2f}")
    print(f'[ALL DONE] elapsed={elapsed:.1f}s')


if __name__ == '__main__':
    args = parse_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_id}'
    main(args)
