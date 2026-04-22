"""
绘制 inputmap（原图）vs scoremap（PromptAD 评分图）vs gt 对比图

用法:
    python plot_scoremap.py                           # 从所有已完成实验中各取1个样本
    python plot_scoremap.py --dataset dsss_signal --scene CaoChang --noise m10db
    python plot_scoremap.py --n-samples 3 --out ./result/scoremap_compare.png
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


TRAIN_SITE   = 'WeaponMuseum_spectrum'
TEST_SITES   = ['Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']
DATASETS     = ['dsss_signal', 'burst_signal', 'chirp_signal']
K_SHOT       = 24


def find_triplets(imgs_dir, n=None):
    """找到 imgs_dir 下所有 (ori, gt, PromptAD) 三元组"""
    if not os.path.isdir(imgs_dir):
        return []
    files = os.listdir(imgs_dir)
    ori_files = sorted(f for f in files if f.endswith('_ori.jpg') and 'abnormal' in f)
    triplets = []
    for ori_f in ori_files:
        stem   = ori_f[:-8]          # remove '_ori.jpg'
        gt_f   = stem + '_gt.jpg'
        score_f = stem + '_PromptAD.jpg'
        if gt_f in files and score_f in files:
            triplets.append((
                os.path.join(imgs_dir, ori_f),
                os.path.join(imgs_dir, gt_f),
                os.path.join(imgs_dir, score_f),
                stem,
            ))
    if n and len(triplets) > n:
        triplets = random.sample(triplets, n)
    return triplets


def collect_samples(root_dir, dataset=None, scene=None, noise=None, n_per_cond=1, seed=111):
    """从跨站点实验（BinBo_to_X）中各取 n_per_cond 个样本"""
    random.seed(seed)
    samples = []   # list of (label, ori_path, gt_path, score_path)

    ds_list = [dataset] if dataset else DATASETS
    sc_list = [scene]   if scene   else TEST_SITES
    nl_list = [noise]   if noise   else NOISE_LEVELS

    for ds in ds_list:
        for sc in sc_list:
            for nl in nl_list:
                imgs_dir = os.path.join(root_dir, ds,
                                        f'{TRAIN_SITE}_to_{sc}', nl,
                                        f'k_{K_SHOT}', 'imgs')
                triplets = find_triplets(imgs_dir, n=n_per_cond)
                for ori, gt, score, stem in triplets:
                    ds_short = ds.replace('_signal', '').upper()
                    label = f'{ds_short} | {TRAIN_SITE}→{sc} | {nl}'
                    samples.append((label, ori, gt, score))
    return samples


def plot_comparison(samples, out_path):
    if not samples:
        print('没有找到任何图像，请先运行实验并确保 --vis True')
        return

    n_rows  = len(samples)
    fig     = plt.figure(figsize=(10, n_rows * 2.8 + 0.5))
    gs      = gridspec.GridSpec(n_rows, 3, figure=fig,
                                wspace=0.03, hspace=0.35)

    col_titles = ['Input', 'Ground Truth', 'Score Map']

    for row, (label, ori_p, gt_p, score_p) in enumerate(samples):
        for col, (img_p, ctitle) in enumerate(zip([ori_p, gt_p, score_p], col_titles)):
            ax = fig.add_subplot(gs[row, col])
            try:
                img = Image.open(img_p)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center',
                        transform=ax.transAxes, fontsize=6, color='red')
            ax.axis('off')
            if row == 0:
                ax.set_title(ctitle, fontsize=10, fontweight='bold', pad=4)
            if col == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=85,
                              va='center', ha='right')

    fig.suptitle(f'Cross-site Anomaly Detection  (Train: {TRAIN_SITE})',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}  ({n_rows} samples)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir',  type=str, default='./result')
    parser.add_argument('--dataset',   type=str, default=None,
                        choices=['dsss_signal', 'burst_signal', 'chirp_signal'])
    parser.add_argument('--scene',     type=str, default=None)
    parser.add_argument('--noise',     type=str, default=None,
                        choices=['m10db', 'm20db', 'm30db'])
    parser.add_argument('--n-samples', type=int, default=1,
                        help='每个条件各取多少个样本')
    parser.add_argument('--seed',      type=int, default=111)
    parser.add_argument('--out',       type=str,
                        default='./result/scoremap_compare.png')
    args = parser.parse_args()

    samples = collect_samples(
        args.root_dir,
        dataset=args.dataset,
        scene=args.scene,
        noise=args.noise,
        n_per_cond=args.n_samples,
        seed=args.seed,
    )
    plot_comparison(samples, args.out)
