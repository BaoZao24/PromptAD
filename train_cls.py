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

import torch.optim.lr_scheduler
import torch
import torch.nn.functional as F
import cv2
import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from utils.visualization import plot_sample_cv2
from torchvision import transforms
import random
from tqdm import tqdm

TASK = 'CLS'


def build_input_transform(input_mode, img_resize, img_cropsize):
    pre_resize_crop = [
        transforms.Resize((img_resize, img_resize), Image.BICUBIC),
        transforms.CenterCrop(img_cropsize),
    ]
    if input_mode == 'gray3':
        return transforms.Compose(pre_resize_crop + [
            Gray3Channels(),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'gray_local2d_edge':
        return transforms.Compose(pre_resize_crop + [
            GrayLocal2DEdgeChannels(win=15),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'spectral_gradient_v2':
        return transforms.Compose(pre_resize_crop + [
            SpectrogramGradientChannelsV2(),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'dsss_weak_residual':
        return transforms.Compose(pre_resize_crop + [
            DSSSWeakResidualChannels(alpha=0.2),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'dsss_gray_weakresidual_weakresidual':
        return transforms.Compose(pre_resize_crop + [
            DSSSGrayWeakResidualResidualChannels(alpha=0.2),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'gray_contrast_only':
        return transforms.Compose(pre_resize_crop + [
            GrayContrastOnlyChannels(),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'weak_residual_only':
        return transforms.Compose(pre_resize_crop + [
            WeakResidualOnlyChannels(alpha=0.2),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    if input_mode == 'grad_mag_only':
        return transforms.Compose(pre_resize_crop + [
            GradMagnitudeOnlyChannels(),
            transforms.Normalize(mean=spectrogram_mean_train, std=spectrogram_std_train),
        ])
    return transforms.Compose(pre_resize_crop + [
        lambda image: image.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train),
    ])


def _to_gray01(raw_img):
    arr = raw_img.numpy() if hasattr(raw_img, 'numpy') else raw_img
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    return gray


def calculate_spectral_stat_score(raw_img, topk_ratio=0.05):
    gray = _to_gray01(raw_img)

    freq_background = np.median(gray, axis=1, keepdims=True)
    residual = np.abs(gray - freq_background)

    local_mean = cv2.blur(gray, (9, 9))
    local_sq_mean = cv2.blur(gray * gray, (9, 9))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    stat_map = 0.5 * residual + 0.5 * local_std
    flat = stat_map.reshape(-1)
    topk = max(1, int(flat.size * topk_ratio))
    return float(np.partition(flat, -topk)[-topk:].mean())


def calculate_batch_stat_scores(raw_data, topk_ratio=0.05):
    return [calculate_spectral_stat_score(img, topk_ratio=topk_ratio) for img in raw_data]


def fuse_with_stat_scores(image_scores, stat_scores, beta):
    return (np.asarray(image_scores, dtype=np.float32) + beta * np.asarray(stat_scores, dtype=np.float32)).tolist()


def select_balanced_visualization_indices(gt_list, normal_count=10, abnormal_count=10):
    normal_indices = [idx for idx, label in enumerate(gt_list) if int(label) == 0]
    abnormal_indices = [idx for idx, label in enumerate(gt_list) if int(label) == 1]

    random.shuffle(normal_indices)
    random.shuffle(abnormal_indices)

    return normal_indices[:min(normal_count, len(normal_indices))] + abnormal_indices[:min(abnormal_count, len(abnormal_indices))]


def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'cnn_mamba_gallery',
        'cnn_mamba_global_gallery',
        'rn50_gallery',
        'rn50_local_gallery',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {
        k: v for k, v in state_dict.items()
        if k in selected_keys or k.startswith('prompt_learner.') or k.startswith('visual_adapters.') or k.startswith('visual_class_prompt_adapter.') or k.startswith('score_fusion_head.') or k.startswith('dense_mask_head.') or k.startswith('text_aligned_dense_head.') or k.startswith('cnn_mamba_local_branch.') or '.lora_' in k
    }

    torch.save(selected_state_dict, path)


def save_image_scores(names, scores_img, gt_list, path, visual_maps=None, text_scores=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = dict(
        names=np.asarray(names),
        scores=np.asarray(scores_img, dtype=np.float32),
        labels=np.asarray(gt_list, dtype=np.int32),
    )
    if visual_maps is not None:
        save_dict['visual_maps'] = np.asarray(visual_maps, dtype=np.float32)
    if text_scores is not None:
        save_dict['text_scores'] = np.asarray(text_scores, dtype=np.float32)
    np.savez_compressed(path, **save_dict)


def save_visualization_samples(names, test_imgs, score_maps, gt_mask_list, gt_list, save_folder):
    normal_indices = [idx for idx, label in enumerate(gt_list) if int(label) == 0]
    abnormal_indices = [idx for idx, label in enumerate(gt_list) if int(label) == 1]

    random.shuffle(normal_indices)
    random.shuffle(abnormal_indices)

    normal_indices = normal_indices[:10]
    abnormal_indices = abnormal_indices[:10]
    selected_indices = normal_indices + abnormal_indices

    if not selected_indices:
        return

    selected_names = [names[idx] for idx in selected_indices]
    selected_imgs = [test_imgs[idx] for idx in selected_indices]
    selected_scores = [score_maps[idx] for idx in selected_indices]
    selected_gts = [gt_mask_list[idx] for idx in selected_indices]

    plot_sample_cv2(
        selected_names,
        selected_imgs,
        {'PromptAD': selected_scores},
        selected_gts,
        save_folder=save_folder,
    )

def rebuild_cnn_mamba_gallery(model, train_data: DataLoader, device: str, desc: str = 'CNN gallery'):
    if model.cnn_mamba_local_branch is None:
        return
    cnn_mamba_features = []
    with torch.no_grad():
        for (data, mask, label, name, img_type) in tqdm(train_data, desc=desc, leave=False):
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)
            cnn_mamba_features.append(model.encode_cnn_mamba_local(data).detach())
    if cnn_mamba_features:
        model.build_cnn_mamba_gallery(torch.cat(cnn_mamba_features, dim=0))


def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        check_path: str,
        train_data: DataLoader,
    csv_path: str,
    img_dir: str,
        ):

    # change the model into eval mode
    model.eval_mode()

    global_features = []
    features1 = []
    features2 = []
    cnn_mamba_features = []
    rn50_features = []
    rn50_local_features = []
    print('Building image feature gallery...')
    for (data, mask, label, name, img_type) in tqdm(train_data, desc='Feature gallery', leave=False):

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
        local_gallery_features = torch.cat(rn50_local_features, dim=0) if rn50_local_features else None
        model.build_rn50_gallery(torch.cat(rn50_features, dim=0), local_gallery_features)

    cached_test_batches = []
    if not model.use_visual_adapter and not model.use_visual_lora:
        # ── Cache test image features once (image encoder is frozen during training) ──
        # Each epoch only text-prompt changes; re-encoding test images is wasteful.
        print('Caching test image features (one-time)...')
        for (raw_data, mask, label, name, img_type) in tqdm(dataloader, desc='Cache test feats', leave=False):
            data_t = [model.transform(Image.fromarray(f.numpy())) for f in raw_data]
            data_t = torch.stack(data_t, dim=0)
            vf = model.encode_image(data_t.to(device))
            stat_scores = None
            if args.stat_fusion:
                stat_scores = calculate_batch_stat_scores(raw_data, topk_ratio=args.stat_topk_ratio)
            cached_test_batches.append({
                'vf':    [v.cpu() for v in vf],         # keep on CPU to save GPU memory
                'mask':  mask,
                'label': label,
                'name':  name,
                'denorm': [denormalization(d.cpu().numpy()) for d in data_t],
                'stat_scores': stat_scores,
                'image': data_t.cpu() if (args.cnn_vit_mamba_fusion or args.rn50_visual_fusion) else None,
            })
        # ─────────────────────────────────────────────────────────────────────────────

    parameter_groups = model.trainable_parameter_groups(
        base_lr=args.lr,
        prompt_lr=args.prompt_lr,
        visual_adapter_lr=args.visual_adapter_lr,
        visual_class_prompt_lr=args.visual_class_prompt_lr,
        score_fusion_lr=args.score_fusion_lr,
        cnn_mamba_lr=args.cnn_mamba_lr,
        visual_lora_lr=args.visual_lora_lr,
        text_aligned_dense_lr=args.text_aligned_dense_lr,
    )
    optimizer = torch.optim.SGD(parameter_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)
    criterion_bce = nn.BCEWithLogitsLoss().to(device)

    best_result_dict = None
    best_visual_maps = None
    best_text_scores = None
    for epoch in range(args.Epoch):
        print(f'Epoch [{epoch + 1}/{args.Epoch}] training...')
        epoch_loss = 0.0
        train_pbar = tqdm(train_data, desc=f'Train {epoch + 1}/{args.Epoch}', leave=False)
        for step, (data, mask, label, name, img_type) in enumerate(train_pbar, start=1):
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)

            # data = data[0:1, :, :, :].to(device)
            data = data.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            visual_features = model.encode_image(data)
            cls_feature = visual_features[0]

            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

            loss_v2t = criterion(logits_v2t, target_v2t)

            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1

            if args.cnn_vit_mamba_fusion and args.cnn_mamba_align_lambda > 0:
                cnn_local_features = model.encode_cnn_mamba_local(data)
                cnn_global_feature = F.normalize(cnn_local_features.mean(dim=1), dim=-1)
                vit_global_target = F.normalize(cls_feature.detach(), dim=-1)
                loss_cnn_align = 1.0 - (cnn_global_feature * vit_global_target).sum(dim=-1).mean()
                loss = loss + args.cnn_mamba_align_lambda * loss_cnn_align

            if args.learnable_score_fusion:
                textual_anomaly = 1.0 - logits_v2t.softmax(dim=-1)[:, 0]
                visual_anomaly_map = model.calculate_visual_anomaly_score(visual_features)
                fusion_logits = model.calculate_learnable_fusion_logit(textual_anomaly, visual_anomaly_map)
                fusion_targets = label.float().to(device).view(-1)
                loss_fusion = criterion_bce(fusion_logits.view(-1), fusion_targets)
                loss = loss + args.learnable_score_fusion_lambda * loss_fusion

            epoch_loss += loss.item()
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{epoch_loss / step:.4f}')

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_text_feature_gallery()
        if model.use_visual_adapter or model.use_visual_lora:
            global_features = []
            features1 = []
            features2 = []
            print('Rebuilding image feature gallery with adapted features...')
            with torch.no_grad():
                for (data, mask, label, name, img_type) in tqdm(train_data, desc='Adapted gallery', leave=False):
                    data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
                    data = torch.stack(data, dim=0).to(device)
                    cls_feature, _, feature_map1, feature_map2 = model.encode_image(data)
                    global_features.append(cls_feature)
                    features1.append(feature_map1)
                    features2.append(feature_map2)
            global_features = torch.cat(global_features, dim=0)
            features1 = torch.cat(features1, dim=0)
            features2 = torch.cat(features2, dim=0)
            model.build_image_feature_gallery(features1, features2, global_features)
            model.set_visual_class_prototype(global_features)

        if model.cnn_mamba_local_branch is not None:
            print('Rebuilding CNN-Mamba gallery with current local branch...')
            rebuild_cnn_mamba_gallery(model, train_data, device, desc='CNN-Mamba gallery')

        print(f'Epoch [{epoch + 1}/{args.Epoch}] evaluating...')
        scores_img = []
        score_maps = []
        visual_maps_raw = []
        text_scores_collected = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        names = []

        if model.use_visual_adapter or model.use_visual_lora:
            with torch.no_grad():
                for (raw_data, mask, label, name, img_type) in tqdm(dataloader, desc=f'Eval {epoch + 1}/{args.Epoch}', leave=False):
                    data_t = [model.transform(Image.fromarray(f.numpy())) for f in raw_data]
                    data_t = torch.stack(data_t, dim=0).to(device)
                    vf_gpu = model.encode_image(data_t)
                    score_img, score_map, raw_maps, ts = model.score_cached(vf_gpu, 'cls', return_raw_map=True, image=data_t if (args.cnn_vit_mamba_fusion or args.rn50_visual_fusion) else None)
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
                    text_scores_collected += list(ts)
        else:
            for batch in tqdm(cached_test_batches, desc=f'Eval {epoch + 1}/{args.Epoch}', leave=False):
                vf_gpu = [v.to(device) for v in batch['vf']]
                score_img, score_map, raw_maps, ts = model.score_cached(vf_gpu, 'cls', return_raw_map=True, image=batch.get('image').to(device) if batch.get('image') is not None else None)
                if args.stat_fusion:
                    score_img = fuse_with_stat_scores(score_img, batch['stat_scores'], args.stat_fusion_beta)

                test_imgs  += batch['denorm']
                names      += list(batch['name'])
                gt_list    += batch['label'].numpy().tolist()
                for m in batch['mask'].numpy():
                    m[m > 0] = 1
                    gt_mask_list.append(m)
                score_maps += score_map
                scores_img += score_img
                visual_maps_raw += raw_maps
                text_scores_collected += list(ts)

        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))

        if args.vis:
            vis_indices = select_balanced_visualization_indices(gt_list, normal_count=10, abnormal_count=10)
            if vis_indices:
                vis_names = [names[idx] for idx in vis_indices]
                vis_imgs = [test_imgs[idx] for idx in vis_indices]
                vis_scores = [score_maps[idx] for idx in vis_indices]
                vis_gts = [gt_mask_list[idx] for idx in vis_indices]
                plot_sample_cv2(vis_names, vis_imgs, {'PromptAD': vis_scores}, vis_gts, save_folder=img_dir)

        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))

        score_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), 'scores', f"Seed_{args.seed}-image_scores.npz")

        if best_result_dict is None:
            best_result_dict = result_dict
            save_check_point(model, check_path)
            save_image_scores(names, scores_img, gt_list, score_path,
                             visual_maps=visual_maps_raw, text_scores=text_scores_collected)
            save_metric(best_result_dict, dataset_classes[args.dataset], args.class_name, args.dataset, csv_path)

        elif best_result_dict['i_roc'] < result_dict['i_roc']:
            best_result_dict = result_dict
            save_check_point(model, check_path)
            save_image_scores(names, scores_img, gt_list, score_path,
                             visual_maps=visual_maps_raw, text_scores=text_scores_collected)
            save_metric(best_result_dict, dataset_classes[args.dataset], args.class_name, args.dataset, csv_path)

        current_i_roc = round(result_dict['i_roc'], 2)
        best_i_roc = round(best_result_dict['i_roc'], 2)
        print(f'Epoch [{epoch + 1}/{args.Epoch}] Image-AUROC: {current_i_roc} | Best: {best_i_roc}')

    return best_result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader, csv_path=csv_path, img_dir=img_dir)

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'spectrum', 'sample', 'deceptive_signal', 'burst_signal', 'dsss_signal', 'chirp_signal', 'pulse_signal', 'wideband_pulse', 'wideband_pulse_png', 'rf_spe_png'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="analysis_outputs/manual_runs")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16', 'ViT-L-14'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=50)
    parser.add_argument("--prompt-mode", type=str, default="rf",
                        choices=["generic", "rf_domain", "rf", "legacy", "rf_object_agnostic",
                                 "rf_scene_conditioned", "rf_signal_structured"],
                        help="generic 使用无频谱语义的通用异常 prompt；rf_domain 只加入 RF/spectrogram 任务域词；rf_signal_structured 使用信号类型结构词")
    parser.add_argument("--text-prototype-mode", type=str, default="single",
                        choices=["single", "grouped_max", "grouped_mean", "grouped_meanmax", "grouped_softmax"],
                        help="single 将所有异常 prompt 平均成一个原型；grouped_max 保留 burst/chirp/dsss 多异常原型并取最大异常分数；grouped_meanmax 使用 0.5*mean + 0.5*max 融合多异常原型分数")
    parser.add_argument("--input-mode", type=str, default="auto",
                        choices=["auto", "rgb", "gray3", "gray_local2d_edge", "morph_fusion_gray_residual_a01", "morph_fusion_local2d_residual_a01", "morph_fusion_tophat_a01", "morph_fusion_multiscale_residual_a01", "morph_fusion_gray_residual_no_contrast_a01", "morph_fusion_clahe_gray"],
                        help="auto 在频谱类数据集上解析为 morph_fusion_gray_residual_a01，其余为 rgb；其它取值为消融/对照方案")
    parser.add_argument("--cls-score-mode", type=str, default="text_only",
                        choices=["text_only", "visual_topk", "visual_topk_max", "visual_topk_freq",
                                 "normal_center", "normal_mahalanobis", "text_normal_center", "text_normal_mahalanobis",
                                 "dense_only", "dense_fusion", "dense_map_only",
                                 "ta_map_only", "ta_image_only", "ta_topk", "ta_max", "ta_max_only",
                                 "ta_norm_excess", "ta_norm_max_excess", "ta_harmonic"],
                        help="图像级分数融合方式")
    parser.add_argument("--visual-topk-ratio", type=float, default=0.05,
                        help="visual patch score 聚合时使用的 top-k 比例")
    parser.add_argument("--visual-gallery-chunk-size", type=int, default=1024,
                        help="visual patch 到 normal gallery 最近邻计算的 gallery 分块大小，调小可降低 eval 显存")
    parser.add_argument("--visual-score-alpha", type=float, default=1.0,
                        help="textual score 融合权重")
    parser.add_argument("--visual-score-beta", type=float, default=1.0,
                        help="visual top-k score 融合权重")
    parser.add_argument("--visual-score-gamma", type=float, default=0.0,
                        help="visual max score 融合权重")
    parser.add_argument("--visual-freq-position-weight", type=float, default=0.0,
                        help="频率位置约束强度，用于强调远离中心或特定频带的异常")
    parser.add_argument("--normal-dist-ridge", type=float, default=1e-4,
                        help="normal distribution diagonal variance 的最小平滑项")
    parser.add_argument("--stat-fusion", type=str2bool, choices=[True, False], default=False,
                        help="是否把传统频谱统计图像级分数直接融合到模型分数中，不做 z-score")
    parser.add_argument("--stat-topk-ratio", type=float, default=0.05,
                        help="统计纹理分数聚合时使用的 top-k 比例")
    parser.add_argument("--stat-fusion-beta", type=float, default=0.5,
                        help="传统频谱统计分数融合权重")
    parser.add_argument("--visual-class-prompt", type=str2bool, choices=[True, False], default=False,
                        help="是否启用 VCPA，将正常视觉原型映射成软 class prompt token")
    parser.add_argument("--visual-class-token-num", type=int, default=2,
                        help="VCPA 生成的软 class token 数量")
    parser.add_argument("--visual-class-prompt-bottleneck-ratio", type=float, default=0.25,
                        help="VCPA 的瓶颈维度比例")
    parser.add_argument("--visual-class-prompt-alpha", type=float, default=0.2,
                        help="VCPA 残差分支权重")
    parser.add_argument("--visual-class-prototype-mode", type=str, default="mean",
                        choices=["mean", "diverse"],
                        help="VCPA 使用的正常视觉原型构造方式：mean 为单均值；diverse 为从正常特征中选多个多样原型")
    parser.add_argument("--visual-class-prototype-num", type=int, default=1,
                        help="visual-class-prototype-mode=diverse 时使用的正常原型数量")
    parser.add_argument("--visual-adapter", type=str2bool, choices=[True, False], default=False,
                        help="是否在冻结 CLIP 视觉特征后训练轻量残差 visual adapter")
    parser.add_argument("--adapter-bottleneck-ratio", type=float, default=0.25,
                        help="visual adapter 的瓶颈维度比例")
    parser.add_argument("--adapter-alpha", type=float, default=0.2,
                        help="visual adapter 残差分支权重")
    parser.add_argument("--visual-lora", type=str2bool, choices=[True, False], default=False,
                        help="是否在 CLIP visual transformer attention 中训练 LoRA")
    parser.add_argument("--visual-lora-rank", type=int, default=4,
                        help="visual LoRA 的低秩维度")
    parser.add_argument("--visual-lora-alpha", type=float, default=8.0,
                        help="visual LoRA 的缩放系数")
    parser.add_argument("--visual-lora-dropout", type=float, default=0.0,
                        help="visual LoRA 的 dropout")
    parser.add_argument("--learnable-score-fusion", type=str2bool, choices=[True, False], default=False,
                        help="是否启用轻量可学习融合头，基于文本分数做残差校正")
    parser.add_argument("--learnable-score-fusion-hidden-dim", type=int, default=4,
                        help="轻量融合头的隐藏维度")
    parser.add_argument("--learnable-score-fusion-alpha", type=float, default=0.25,
                        help="轻量融合头残差项的缩放系数")
    parser.add_argument("--learnable-score-fusion-lambda", type=float, default=0.5,
                        help="轻量融合头辅助 BCE 损失的权重")
    parser.add_argument("--text-aligned-dense", type=str2bool, choices=[True, False], default=False,
                        help="是否启用 APRIL-GAN 风格 visual->text projection dense 分支")
    parser.add_argument("--text-aligned-dense-score-beta", type=float, default=1.0,
                        help="ta_* 模式下 text-aligned dense 图像分数的融合权重")
    parser.add_argument("--ta-normal-quantile", type=float, default=0.95,
                        help="ta_norm_excess 使用的 target normal 分位数阈值")
    parser.add_argument("--ta-normal-excess-scale-floor", type=float, default=1e-6,
                        help="ta_norm_excess 归一化尺度下限，避免除零")
    parser.add_argument("--rn50-visual-fusion", type=str2bool, choices=[True, False], default=False,
                        help="是否启用冻结 CLIP-RN50 视觉距离分支，与主 ViT/prompt 分数保守融合")
    parser.add_argument("--rn50-pretrained", type=str, default="openai",
                        help="RN50 分支使用的 CLIP 预训练权重名称，默认 openai")
    parser.add_argument("--rn50-visual-beta", type=float, default=0.05,
                        help="RN50 正常特征距离分数的融合权重")
    parser.add_argument("--rn50-score-mode", type=str, default="global", choices=["global", "local_topk", "both"],
                        help="RN50 辅助分数模式：global 用最终 embedding，local_topk 用 layer4 局部 token，both 两者平均")
    parser.add_argument("--rn50-fusion-mode", type=str, default="score", choices=["score", "guided_vit"],
                        help="score 表示 RN50 自己出辅助分数；guided_vit 表示 RN50 local map 引导 ViT patch map 聚合")
    parser.add_argument("--rn50-local-topk-ratio", type=float, default=0.1,
                        help="RN50 local_topk 模式下聚合最高异常 patch 的比例")
    parser.add_argument("--rn50-guidance-temperature", type=float, default=0.2,
                        help="guided_vit 模式下 RN50 attention softmax 温度，越小越集中关注高分局部区域")
    parser.add_argument("--cnn-vit-mamba-fusion", type=str2bool, choices=[True, False], default=False,
                        help="是否启用 CNN + ViT + Mamba 风格轻量融合头")
    parser.add_argument("--cnn-mamba-beta", type=float, default=None,
                        help="CNN-Mamba local image score 的保守融合权重")
    parser.add_argument("--cnn-mamba-map-beta", type=float, default=None,
                        help="CNN-Mamba local heatmap 融合权重，默认不影响原 patch map")
    parser.add_argument("--cnn-mamba-pool-topk-ratio", type=float, default=0.2,
                        help="CNN token 聚合时只关注异常距离最高的局部区域比例")
    parser.add_argument("--cnn-mamba-pool-temperature", type=float, default=0.1,
                        help="CNN token 异常加权聚合的 softmax 温度，越小越集中关注高分区域")
    parser.add_argument("--cnn-mamba-align-lambda", type=float, default=0.0,
                        help="正常样本上 CNN 聚合特征与 CLIP-ViT 全局特征的对齐损失权重")
    parser.add_argument("--cnn-vit-mamba-alpha", type=float, default=0.05,
                        help="兼容旧命令：等价于 --cnn-mamba-beta")
    parser.add_argument("--cnn-vit-mamba-patch-alpha", type=float, default=0.0,
                        help="兼容旧命令：等价于 --cnn-mamba-map-beta")
    parser.add_argument("--cnn-vit-mamba-num-blocks", type=int, default=2,
                        help="CNN+ViT+Mamba token mixer 堆叠层数")
    parser.add_argument("--cnn-vit-mamba-dropout", type=float, default=0.0,
                        help="CNN+ViT+Mamba token mixer dropout")
    parser.add_argument("--prompt-lr", type=float, default=None,
                        help="prompt learner 的单独学习率；默认使用 --lr")
    parser.add_argument("--visual-class-prompt-lr", type=float, default=None,
                        help="VCPA 的单独学习率；默认使用 --lr")
    parser.add_argument("--visual-adapter-lr", type=float, default=None,
                        help="visual adapter 的单独学习率；默认使用 --lr")
    parser.add_argument("--score-fusion-lr", type=float, default=None,
                        help="score_img 融合头的单独学习率；默认使用 --lr")
    parser.add_argument("--cnn-mamba-lr", type=float, default=None,
                        help="CNN-Mamba 分支的单独学习率；默认使用 --lr")
    parser.add_argument("--visual-lora-lr", type=float, default=None,
                        help="visual LoRA 的单独学习率；默认使用 --lr")
    parser.add_argument("--text-aligned-dense-lr", type=float, default=None,
                        help="text-aligned dense projection 分支的单独学习率；默认使用 --lr")
    # burst_signal related
    parser.add_argument("--noise-level", type=str, default='m10db', choices=['m10db', 'm20db', 'm30db', 'm40db', 'm50db', 'm55db', 'm60db', 'm70db'])

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)

    args = parser.parse_args()
    if args.backbone == "ViT-L-14":
        # ViT-L-14 uses a 14px patch and the bundled 224px positional grid.
        if args.img_resize == 240:
            args.img_resize = 224
        if args.img_cropsize == 240:
            args.img_cropsize = 224

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
