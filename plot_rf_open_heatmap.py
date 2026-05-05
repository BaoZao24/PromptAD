"""
RF Open 实验结果热力图 + 文字表格

行：信号类型  [Burst, Chirp, DSSS]
列：JSR 级别  [−10 dB, −20 dB, −30 dB]
值：Image-AUROC

CSV 路径格式：
  {root_dir}/rf_open/{signal}_{jsr}/k_{k_shot}/csv/Seed_{seed}-results.csv

用法:
    python plot_rf_open_heatmap.py
    python plot_rf_open_heatmap.py --root-dir ./result --k-shot 1 --seed 111 --out rf_open_heatmap.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SIGNAL_TYPES  = ['burst', 'chirp', 'dsss']
NOISE_LEVELS  = ['m10db', 'm20db', 'm30db']

ROW_LABELS    = ['Burst', 'Chirp', 'DSSS']
COL_LABELS    = ['−10 dB', '−20 dB', '−30 dB']
DATASET       = 'rf_open'


def read_i_roc(root_dir: str, signal: str, jsr: str, k_shot: int, seed: int) -> float:
    class_name = f'{signal}_{jsr}'
    csv_path = os.path.join(
        root_dir, DATASET, class_name, f'k_{k_shot}', 'csv',
        f'Seed_{seed}-results.csv',
    )
    try:
        df  = pd.read_csv(csv_path, index_col=0)
        key = f'{DATASET}-{class_name}'
        val = float(df.loc[key, 'i_roc'])
        return val / 100.0 if val > 1.0 else val   # 统一到 [0, 1]
    except Exception:
        return np.nan


def build_matrix(root_dir: str, k_shot: int, seed: int) -> np.ndarray:
    mat = np.full((len(SIGNAL_TYPES), len(NOISE_LEVELS)), np.nan)
    for i, sig in enumerate(SIGNAL_TYPES):
        for j, jsr in enumerate(NOISE_LEVELS):
            mat[i, j] = read_i_roc(root_dir, sig, jsr, k_shot, seed)
    return mat


def print_table(mat: np.ndarray):
    col_w = 10
    header = f'{"Signal":<8}' + ''.join(f'{c:>{col_w}}' for c in COL_LABELS) + f'{"Mean":>{col_w}}'
    sep    = '-' * len(header)
    print(f'\n{"="*60}')
    print(f'RF Open — Image-AUROC')
    print(f'{"="*60}')
    print(header)
    print(sep)

    col_vals = [[] for _ in NOISE_LEVELS]
    for i, sig in enumerate(SIGNAL_TYPES):
        row  = f'{ROW_LABELS[i]:<8}'
        row_vals = []
        for j in range(len(NOISE_LEVELS)):
            v = mat[i, j]
            if not np.isnan(v):
                col_vals[j].append(v)
                row_vals.append(v)
                row += f'{v:>{col_w}.4f}'
            else:
                row += f'{"N/A":>{col_w}}'
        mean_v = np.mean(row_vals) if row_vals else float('nan')
        row += f'{mean_v:>{col_w}.4f}' if not np.isnan(mean_v) else f'{"N/A":>{col_w}}'
        print(row)

    print(sep)
    mean_row = f'{"Mean":<8}'
    all_vals = []
    for j in range(len(NOISE_LEVELS)):
        vals = col_vals[j]
        mean_row += f'{np.mean(vals):>{col_w}.4f}' if vals else f'{"N/A":>{col_w}}'
        all_vals.extend(vals)
    overall = np.mean(all_vals) if all_vals else float('nan')
    mean_row += f'{overall:>{col_w}.4f}' if not np.isnan(overall) else f'{"N/A":>{col_w}}'
    print(mean_row)
    print()


def save_csv(mat: np.ndarray, csv_path: str):
    rows = []
    for i, sig in enumerate(SIGNAL_TYPES):
        row = {'signal': ROW_LABELS[i]}
        row_vals = []
        for j, nl in enumerate(NOISE_LEVELS):
            v = mat[i, j]
            row[nl] = round(float(v), 4) if not np.isnan(v) else None
            if not np.isnan(v):
                row_vals.append(v)
        row['mean'] = round(float(np.mean(row_vals)), 4) if row_vals else None
        rows.append(row)

    # mean row
    mean_row = {'signal': 'Mean'}
    all_vals = []
    for j, nl in enumerate(NOISE_LEVELS):
        col_vals = [mat[i, j] for i in range(len(SIGNAL_TYPES)) if not np.isnan(mat[i, j])]
        mean_row[nl] = round(float(np.mean(col_vals)), 4) if col_vals else None
        all_vals.extend(col_vals)
    mean_row['mean'] = round(float(np.mean(all_vals)), 4) if all_vals else None
    rows.append(mean_row)

    df = pd.DataFrame(rows).set_index('signal')
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path)
    print(f'Saved summary CSV → {csv_path}')


def plot_heatmap(root_dir: str, k_shot: int, seed: int, out_path: str):
    mat = build_matrix(root_dir, k_shot, seed)

    print_table(mat)

    csv_path = out_path.replace('.png', '.csv')
    save_csv(mat, csv_path)

    fig, ax = plt.subplots(figsize=(6, 3.6))

    vmin = max(0.4, np.nanmin(mat) - 0.05) if not np.all(np.isnan(mat)) else 0.4
    vmax = min(1.0, np.nanmax(mat) + 0.05) if not np.all(np.isnan(mat)) else 1.0

    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap='RdYlGn', aspect='auto')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'black' if vmin + 0.15 < v < vmax - 0.15 else 'white'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=9, color='gray')

    ax.set_xticks(range(len(NOISE_LEVELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(SIGNAL_TYPES)))
    ax.set_yticklabels(ROW_LABELS, fontsize=10, fontweight='bold')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Image-AUROC', fontsize=9)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ax.set_title(
        f'RF Open — Anomaly Detection (k-shot={k_shot}, seed={seed})',
        fontsize=11, pad=18,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved heatmap → {out_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str,  default='./result')
    parser.add_argument('--k-shot',   type=int,  default=1)
    parser.add_argument('--seed',     type=int,  default=111)
    parser.add_argument('--out',      type=str,  default='./result/rf_open_heatmap.png')
    args = parser.parse_args()

    plot_heatmap(args.root_dir, args.k_shot, args.seed, args.out)
