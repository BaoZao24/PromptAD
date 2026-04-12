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
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

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

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
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

        print(f'Epoch [{epoch + 1}/{args.Epoch}] evaluating...')
        scores_img = []
        score_maps = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        names = []

        for (data, mask, label, name, img_type) in tqdm(dataloader, desc=f'Eval {epoch + 1}/{args.Epoch}', leave=False):

            data = [model.transform(Image.fromarray(f.numpy())) for f in data]
            data = torch.stack(data, dim=0)

            for d, n, l, m in zip(data, name, label, mask):
                test_imgs += [denormalization(d.cpu().numpy())]
                l = l.numpy()
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_list += [l]
                gt_mask_list += [m]

            data = data.to(device)
            score_img, score_map = model(data, 'cls')
            score_maps += score_map
            scores_img += score_img

        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))

        if args.vis or args.dataset == 'dcase2025':
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
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'spectrum', 'sample', 'dcase2025', 'deceptive'])
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
    parser.add_argument("--Epoch", type=int, default=10)

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
