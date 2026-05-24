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
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {
        k: v for k, v in state_dict.items()
        if k in selected_keys or k.startswith('visual_adapters.') or '.lora_' in k
    }

    torch.save(selected_state_dict, path)


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

    features1 = []
    features2 = []
    print('Building image feature gallery...')
    for (data, mask, label, name, img_type) in tqdm(train_data, desc='Feature gallery', leave=False):

        data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
        data = torch.stack(data, dim=0).to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)

    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)

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
            })
        # ─────────────────────────────────────────────────────────────────────────────

    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
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

            cls_feature, _, _, _ = model.encode_image(data)

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

            epoch_loss += loss.item()
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{epoch_loss / step:.4f}')

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_text_feature_gallery()
        if model.use_visual_adapter or model.use_visual_lora:
            features1 = []
            features2 = []
            print('Rebuilding image feature gallery with adapted features...')
            with torch.no_grad():
                for (data, mask, label, name, img_type) in tqdm(train_data, desc='Adapted gallery', leave=False):
                    data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
                    data = torch.stack(data, dim=0).to(device)
                    _, _, feature_map1, feature_map2 = model.encode_image(data)
                    features1.append(feature_map1)
                    features2.append(feature_map2)
            features1 = torch.cat(features1, dim=0)
            features2 = torch.cat(features2, dim=0)
            model.build_image_feature_gallery(features1, features2)

        print(f'Epoch [{epoch + 1}/{args.Epoch}] evaluating...')
        scores_img = []
        score_maps = []
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
                    score_img, score_map = model.score_cached(vf_gpu, 'cls')
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
        else:
            for batch in tqdm(cached_test_batches, desc=f'Eval {epoch + 1}/{args.Epoch}', leave=False):
                vf_gpu = [v.to(device) for v in batch['vf']]
                score_img, score_map = model.score_cached(vf_gpu, 'cls')
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

        if best_result_dict is None:
            best_result_dict = result_dict
            save_check_point(model, check_path)
            save_metric(best_result_dict, dataset_classes[args.dataset], args.class_name, args.dataset, csv_path)

        elif best_result_dict['i_roc'] < result_dict['i_roc']:
            best_result_dict = result_dict
            save_check_point(model, check_path)
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
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'spectrum', 'sample', 'deceptive_signal', 'burst_signal', 'dsss_signal', 'chirp_signal', 'wideband_pulse', 'wideband_pulse_png', 'rf_spe_png'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
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
    parser.add_argument("--input-mode", type=str, default="auto",
                        choices=["auto", "rgb", "spectral_gradient", "spectral_gradient_v2", "chirp_directional", "chirp_ridge", "chirp_track_enhance", "chirp_rgb_track", "signal_adaptive", "signal_adaptive_v2", "log_power",
                                 "dsss_statistical", "dsss_energy_smooth", "dsss_lowfreq_band", "dsss_energy_profile",
                                 "dsss_rgb_residual", "dsss_weak_residual", "dsss_clahe"],
                        help="signal_adaptive 对 burst/chirp 使用频谱梯度、对 dsss 使用 RGB；signal_adaptive_v2 对 dsss 使用 weak residual；log_power 使用灰度 log-power 压缩")
    parser.add_argument("--cls-score-mode", type=str, default="text_only",
                        choices=["text_only", "visual_topk", "visual_topk_max", "visual_topk_freq"],
                        help="图像级分数融合方式")
    parser.add_argument("--visual-topk-ratio", type=float, default=0.05,
                        help="visual patch score 聚合时使用的 top-k 比例")
    parser.add_argument("--visual-score-alpha", type=float, default=1.0,
                        help="textual score 融合权重")
    parser.add_argument("--visual-score-beta", type=float, default=1.0,
                        help="visual top-k score 融合权重")
    parser.add_argument("--visual-score-gamma", type=float, default=0.0,
                        help="visual max score 融合权重")
    parser.add_argument("--visual-freq-position-weight", type=float, default=0.0,
                        help="频率位置约束强度，用于强调远离中心或特定频带的异常")
    parser.add_argument("--stat-fusion", type=str2bool, choices=[True, False], default=False,
                        help="是否把传统频谱统计图像级分数直接融合到模型分数中，不做 z-score")
    parser.add_argument("--stat-topk-ratio", type=float, default=0.05,
                        help="统计纹理分数聚合时使用的 top-k 比例")
    parser.add_argument("--stat-fusion-beta", type=float, default=0.5,
                        help="传统频谱统计分数融合权重")
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
    parser.add_argument("--split-mode", type=str, default="legacy", choices=["legacy", "normal_75_25"],
                        help="legacy 使用原始 few-shot 切分；normal_75_25 使用 3/4 normal 训练、1/4 normal 测试")
    parser.add_argument("--normal-train-ratio", type=float, default=0.75,
                        help="normal_75_25 模式下正常样本训练比例")

    # burst_signal related
    parser.add_argument("--noise-level", type=str, default='m10db', choices=['m10db', 'm20db', 'm30db', 'm40db', 'm50db'])

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
