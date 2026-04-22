"""
绘制跨站点评估结果热力图

行：(test_site, jsr) 的所有组合
列：各信号类型 [dsss_signal, burst_signal, chirp_signal]
值：i_roc (Image-AUROC)

用法:
    python plot_heatmap.py
    python plot_heatmap.py --root-dir ./result --seed 111 --out heatmap.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_SITE  = 'WeaponMuseum_spectrum'
TEST_SITES  = ['Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']
DATASETS    = ['dsss_signal', 'burst_signal', 'chirp_signal']
K_SHOT      = 24

COL_LABELS  = ['DSSS', 'Burst', 'Chirp']
SITE_SHORT  = {'Playground_spectrum': 'PG', 'TimeSquare_spectrum': 'TS', 'Gymnasium_spectrum': 'GYM'}
JSR_SHORT   = {'m10db': '−10dB', 'm20db': '−20dB', 'm30db': '−30dB'}


def read_i_roc(root_dir, dataset, train_site, test_scene, noise_level, seed):
    dir_class = f'{train_site}_to_{test_scene}'
    csv_path  = os.path.join(root_dir, dataset, dir_class, noise_level,
                             f'k_{K_SHOT}', 'csv', f'Seed_{seed}-results.csv')
    try:
        df  = pd.read_csv(csv_path, index_col=0)
        key = f'{dataset}-{test_scene}'
        val = float(df.loc[key, 'i_roc'])
        return val / 100.0 if val > 1.0 else val   # 统一到 [0,1]
    except Exception:
        return np.nan


def build_matrix(root_dir, seed):
    row_keys = [(site, jsr) for site in TEST_SITES for jsr in NOISE_LEVELS]
    mat = np.full((len(row_keys), len(DATASETS)), np.nan)
    for i, (site, jsr) in enumerate(row_keys):
        for j, ds in enumerate(DATASETS):
            mat[i, j] = read_i_roc(root_dir, ds, TRAIN_SITE, site, jsr, seed)
    return mat, row_keys


def plot_heatmap(root_dir, seed, out_path):
    mat, row_keys = build_matrix(root_dir, seed)

    row_labels = [f'{SITE_SHORT[s]}\n{JSR_SHORT[jsr]}' for s, jsr in row_keys]

    fig, ax = plt.subplots(figsize=(len(DATASETS) * 1.8 + 1.5, len(row_keys) * 0.55 + 1.5))

    im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap='RdYlGn', aspect='auto')

    # 格内写数值
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'black' if 0.55 < v < 0.90 else 'white'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=7, color='gray')

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(COL_LABELS, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # 分隔线：每 3 行（同 site 内的 JSR）之间加细线
    for k in range(1, len(TEST_SITES)):
        ax.axhline(k * len(NOISE_LEVELS) - 0.5, color='white', linewidth=2)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Image-AUROC')
    ax.set_title(f'Cross-site Anomaly Detection  (Train: {TRAIN_SITE}, k-shot={K_SHOT})',
                 fontsize=11, pad=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    # 也打印文字表格
    print('\n' + '='*60)
    header = f'{"Site":<6} {"JSR":<8}' + ''.join(f'{c:>10}' for c in COL_LABELS)
    print(header)
    print('-' * len(header))
    for i, (site, jsr) in enumerate(row_keys):
        row = f'{SITE_SHORT[site]:<6} {JSR_SHORT[jsr]:<8}'
        for j in range(len(DATASETS)):
            v = mat[i, j]
            row += f'{v:>10.4f}' if not np.isnan(v) else f'{"N/A":>10}'
        print(row)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./result')
    parser.add_argument('--seed',     type=int, default=111)
    parser.add_argument('--out',      type=str, default='./result/heatmap_cross_site.png')
    args = parser.parse_args()

    plot_heatmap(args.root_dir, args.seed, args.out)
