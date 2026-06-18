import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import dataset_classes, denormalization, get_dataloader_from_args
from PromptAD import PromptAD
from PromptAD.model import NormalBgDeviationChannels
from train_cls import (
    TripletLoss,
    calculate_batch_stat_scores,
    fuse_with_stat_scores,
    get_args as get_base_args,
    rebuild_cnn_mamba_gallery,
    save_check_point,
    save_image_scores,
    select_balanced_visualization_indices,
    str2bool,
)
from utils.csv_utils import save_metric
from utils.eval_utils import specify_resolution
from utils.metrics import metric_cal_img
from utils.training_utils import setup_seed
from utils.visualization import plot_sample_cv2

TASK = 'CROSS_CLS'


def get_cross_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-dataset', type=str, default=None,
                        choices=['burst_signal', 'chirp_signal', 'dsss_signal', 'wideband_pulse', 'wideband_pulse_png', 'rf_spe_png'])
    parser.add_argument('--target-dataset', type=str, default=None,
                        choices=['burst_signal', 'chirp_signal', 'dsss_signal', 'wideband_pulse', 'wideband_pulse_png', 'rf_spe_png'])
    parser.add_argument('--source-class-name', type=str, default=None)
    parser.add_argument('--target-class-name', type=str, default=None)
    parser.add_argument('--source-noise-level', type=str, default=None)
    parser.add_argument('--target-noise-level', type=str, default=None)
    parser.add_argument('--cross-root-dir', type=str, default=None)
    parser.add_argument('--supervised-triplet', type=str2bool, choices=[True, False], default=True,
                        help='跨库训练时只在 source normal query 上保留 triplet loss')
    cross_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_base_args()
    finally:
        sys.argv = original_argv

    args.source_dataset = cross_args.source_dataset or args.dataset
    args.target_dataset = cross_args.target_dataset or args.dataset
    args.source_class_name = cross_args.source_class_name or args.class_name
    args.target_class_name = cross_args.target_class_name or args.class_name
    args.source_noise_level = cross_args.source_noise_level or args.noise_level
    args.target_noise_level = cross_args.target_noise_level or args.noise_level
    args.cross_root_dir = cross_args.cross_root_dir or args.root_dir
    args.supervised_triplet = cross_args.supervised_triplet
    return args


def make_loader_args(args, dataset, class_name, noise_level):
    kwargs = vars(args).copy()
    kwargs['dataset'] = dataset
    kwargs['class_name'] = class_name
    kwargs['noise_level'] = noise_level
    return kwargs


def get_cross_dirs(args):
    source = f'{args.source_dataset}_{args.source_noise_level}'
    target = f'{args.target_dataset}_{args.target_noise_level}'
    class_pair = f'{args.source_class_name}_to_{args.target_class_name}'
    mode = f'{args.input_mode}_{args.prompt_mode}_{args.cls_score_mode}'
    root = os.path.join(
        args.cross_root_dir,
        'cross_cls',
        f'{source}_to_{target}',
        class_pair,
        args.split_mode,
        mode,
        f'k_{args.k_shot}',
    )
    img_dir = os.path.join(root, 'imgs')
    csv_dir = os.path.join(root, 'csv')
    check_dir = os.path.join(root, 'checkpoint')
    score_dir = os.path.join(root, 'scores')
    for path in (img_dir, csv_dir, check_dir, score_dir):
        os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'Seed_{args.seed}-results.csv')
    check_path = os.path.join(
        check_dir,
        f'{TASK}-Seed_{args.seed}-{source}-to-{target}-{class_pair}-check_point.pt',
    )
    score_path = os.path.join(score_dir, f'Seed_{args.seed}-image_scores.npz')
    return img_dir, csv_path, check_path, score_path


@torch.no_grad()
def build_normal_gallery(model, loader: DataLoader, args, device: str, desc: str):
    model.eval_mode()
    if 'normal_bg' in model.input_mode:
        median, mad = NormalBgDeviationChannels.compute_normal_bg_stats_from_dataloader(
            loader, args.img_resize, args.img_cropsize)
        model.set_normal_bg_stats(median, mad)

    global_features = []
    features1 = []
    features2 = []
    cnn_mamba_features = []
    rn50_features = []
    rn50_local_features = []
    for data, mask, label, name, img_type in tqdm(loader, desc=desc, leave=False):
        data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
        data = torch.stack(data, dim=0).to(device)
        cls_feature, _, feature_map1, feature_map2 = model.encode_image(data)
        global_features.append(cls_feature)
        features1.append(feature_map1)
        features2.append(feature_map2)
        if model.cnn_mamba_local_branch is not None:
            cnn_mamba_features.append(model.encode_cnn_mamba_local(data).detach())
        if model.rn50_model is not None:
            rn50_features.append(model.encode_rn50_image(data).detach())
            if model.rn50_score_mode in {'local_topk', 'both'} or model.rn50_fusion_mode == 'guided_vit':
                rn50_local_features.append(model.encode_rn50_local(data).detach())

    global_features = torch.cat(global_features, dim=0)
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2, global_features)
    model.set_visual_class_prototype(global_features)
    if model.cnn_mamba_local_branch is not None and cnn_mamba_features:
        model.build_cnn_mamba_gallery(torch.cat(cnn_mamba_features, dim=0))
    if model.rn50_model is not None and rn50_features:
        local_gallery = torch.cat(rn50_local_features, dim=0) if rn50_local_features else None
        model.build_rn50_gallery(torch.cat(rn50_features, dim=0), local_gallery)


def train_one_epoch(model, source_query_loader: DataLoader, args, device: str, epoch_idx: int):
    model.train()
    optimizer = train_one_epoch.optimizer
    criterion = train_one_epoch.criterion
    criterion_tip = train_one_epoch.criterion_tip
    criterion_bce = train_one_epoch.criterion_bce

    epoch_loss = 0.0
    train_pbar = tqdm(source_query_loader, desc=f'Source Train {epoch_idx}/{args.Epoch}', leave=False)
    for step, (data, mask, label, name, img_type) in enumerate(train_pbar, start=1):
        data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
        data = torch.stack(data, dim=0).to(device)
        label = label.to(device).long().view(-1)

        normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()
        optimizer.zero_grad()

        normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)
        abnormal_text_features_handle = model.encode_text_embedding(
            abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
        abnormal_text_features_learned = model.encode_text_embedding(
            abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
        abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

        mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
        mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
        loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

        visual_features = model.encode_image(data)
        cls_feature = visual_features[0]

        normal_anchor = F.normalize(normal_text_features.mean(dim=0, keepdim=True), dim=-1)
        abnormal_anchor = F.normalize(abnormal_text_features.mean(dim=0, keepdim=True), dim=-1)
        abnormal_text_features = F.normalize(abnormal_text_features, dim=-1)

        l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_anchor.transpose(0, 1))
        l_neg = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1)).max(dim=1, keepdim=True)[0]
        logit_scale = model.model.logit_scale.half() if model.precision == 'fp16' else model.model.logit_scale
        logits_v2t = torch.cat([l_pos, l_neg], dim=-1) * logit_scale
        loss = criterion(logits_v2t, label.clamp(0, 1)) + loss_match_abnormal * args.lambda1

        if args.supervised_triplet:
            normal_mask = label == 0
            if normal_mask.any():
                loss = loss + criterion_tip(cls_feature[normal_mask], normal_anchor, abnormal_anchor)

        if args.learnable_score_fusion:
            with torch.no_grad():
                textual_anomaly = torch.as_tensor(
                    model.calculate_textual_anomaly_score(visual_features, 'cls'),
                    device=device,
                    dtype=visual_features[0].dtype,
                )
            visual_anomaly_map = model.calculate_visual_anomaly_score(visual_features)
            fusion_logits = model.calculate_learnable_fusion_logit(textual_anomaly, visual_anomaly_map)
            loss = loss + args.learnable_score_fusion_lambda * criterion_bce(fusion_logits.view(-1), label.float())

        if args.cnn_vit_mamba_fusion and args.cnn_mamba_align_lambda > 0:
            cnn_local_features = model.encode_cnn_mamba_local(data)
            cnn_global_feature = F.normalize(cnn_local_features.mean(dim=1), dim=-1)
            vit_global_target = F.normalize(cls_feature.detach(), dim=-1)
            loss_cnn_align = 1.0 - (cnn_global_feature * vit_global_target).sum(dim=-1).mean()
            loss = loss + args.cnn_mamba_align_lambda * loss_cnn_align

        epoch_loss += loss.item()
        train_pbar.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{epoch_loss / step:.4f}')
        loss.backward()
        optimizer.step()


def evaluate_target(model, target_test_loader: DataLoader, args, device: str, epoch_idx: int):
    scores_img = []
    score_maps = []
    visual_maps_raw = []
    text_scores_collected = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    model.eval_mode()
    with torch.no_grad():
        for raw_data, mask, label, name, img_type in tqdm(target_test_loader, desc=f'Target Eval {epoch_idx}/{args.Epoch}', leave=False):
            data_t = [model.transform(Image.fromarray(f.numpy())) for f in raw_data]
            data_t = torch.stack(data_t, dim=0).to(device)
            visual_features = model.encode_image(data_t)
            score_img, score_map, raw_maps, text_scores = model.score_cached(
                visual_features,
                'cls',
                return_raw_map=True,
                image=data_t if (args.cnn_vit_mamba_fusion or args.rn50_visual_fusion) else None,
            )
            if args.stat_fusion:
                stat_scores = calculate_batch_stat_scores(raw_data, topk_ratio=args.stat_topk_ratio)
                score_img = fuse_with_stat_scores(score_img, stat_scores, args.stat_fusion_beta)

            test_imgs += [denormalization(d.cpu().numpy()) for d in data_t]
            names += list(name)
            gt_list += label.numpy().tolist()
            for m in mask.numpy():
                m[m > 0] = 1
                gt_mask_list.append(m)
            score_maps += score_map
            scores_img += score_img
            visual_maps_raw += raw_maps
            text_scores_collected += list(text_scores)

    test_imgs, score_maps, gt_mask_list = specify_resolution(
        test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
    return scores_img, score_maps, test_imgs, gt_list, gt_mask_list, names, visual_maps_raw, text_scores_collected


def fit_cross(model, args, source_gallery_loader, source_query_loader, target_gallery_loader,
              target_test_loader, device, check_path, csv_path, img_dir, score_path):
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    train_one_epoch.optimizer = optimizer
    train_one_epoch.criterion = nn.CrossEntropyLoss().to(device)
    train_one_epoch.criterion_tip = TripletLoss(margin=0.0)
    train_one_epoch.criterion_bce = nn.BCEWithLogitsLoss().to(device)

    best_result_dict = None
    for epoch in range(args.Epoch):
        print(f'Epoch [{epoch + 1}/{args.Epoch}] building source gallery...')
        build_normal_gallery(model, source_gallery_loader, args, device, desc='Source gallery')
        model.build_text_feature_gallery()

        print(f'Epoch [{epoch + 1}/{args.Epoch}] source supervised training...')
        train_one_epoch(model, source_query_loader, args, device, epoch + 1)
        scheduler.step()

        if model.cnn_mamba_local_branch is not None:
            rebuild_cnn_mamba_gallery(model, source_gallery_loader, device, desc='Source CNN-Mamba gallery')

        print(f'Epoch [{epoch + 1}/{args.Epoch}] building target gallery...')
        build_normal_gallery(model, target_gallery_loader, args, device, desc='Target gallery')
        model.build_text_feature_gallery()

        print(f'Epoch [{epoch + 1}/{args.Epoch}] target evaluating...')
        scores_img, score_maps, test_imgs, gt_list, gt_mask_list, names, visual_maps_raw, text_scores = evaluate_target(
            model, target_test_loader, args, device, epoch + 1)

        if args.vis:
            vis_indices = select_balanced_visualization_indices(gt_list, normal_count=10, abnormal_count=10)
            if vis_indices:
                plot_sample_cv2(
                    [names[idx] for idx in vis_indices],
                    [test_imgs[idx] for idx in vis_indices],
                    {'PromptAD': [score_maps[idx] for idx in vis_indices]},
                    [gt_mask_list[idx] for idx in vis_indices],
                    save_folder=img_dir,
                )

        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))
        if best_result_dict is None or best_result_dict['i_roc'] < result_dict['i_roc']:
            best_result_dict = result_dict
            save_check_point(model, check_path)
            save_image_scores(names, scores_img, gt_list, score_path,
                              visual_maps=visual_maps_raw, text_scores=text_scores)
            save_metric(best_result_dict, dataset_classes[args.target_dataset], args.target_class_name,
                        args.target_dataset, csv_path)

        print(f"Epoch [{epoch + 1}/{args.Epoch}] Target Image-AUROC: {round(result_dict['i_roc'], 2)} | Best: {round(best_result_dict['i_roc'], 2)}")

    return best_result_dict


def main(args):
    if args.seed is None:
        args.seed = 111
    setup_seed(args.seed)

    device = 'cuda:0' if args.use_cpu == 0 else 'cpu'
    args.device = device
    args.out_size_h = args.resolution
    args.out_size_w = args.resolution

    source_kwargs = make_loader_args(args, args.source_dataset, args.source_class_name, args.source_noise_level)
    target_kwargs = make_loader_args(args, args.target_dataset, args.target_class_name, args.target_noise_level)

    source_gallery_loader, _ = get_dataloader_from_args(phase='train', perturbed=False, **source_kwargs)
    source_query_loader, _ = get_dataloader_from_args(phase='test', perturbed=False, **source_kwargs)
    target_gallery_loader, _ = get_dataloader_from_args(phase='train', perturbed=False, **target_kwargs)
    target_test_loader, _ = get_dataloader_from_args(phase='test', perturbed=False, **target_kwargs)

    model_kwargs = vars(args).copy()
    model_kwargs['dataset'] = args.source_dataset
    model_kwargs['class_name'] = args.source_class_name
    model_kwargs['noise_level'] = args.source_noise_level

    img_dir, csv_path, check_path, score_path = get_cross_dirs(args)
    model = PromptAD(**model_kwargs).to(device)

    metrics = fit_cross(
        model,
        args,
        source_gallery_loader,
        source_query_loader,
        target_gallery_loader,
        target_test_loader,
        device,
        check_path,
        csv_path,
        img_dir,
        score_path,
    )
    print(
        f"Cross:{args.source_dataset}/{args.source_class_name}/{args.source_noise_level} -> "
        f"{args.target_dataset}/{args.target_class_name}/{args.target_noise_level} "
        f"Image-AUROC:{round(metrics['i_roc'], 2)}"
    )


if __name__ == '__main__':
    args = get_cross_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_id}'
    main(args)
