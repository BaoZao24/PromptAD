"""
RF Open 公开数据集批量实验：
  - 信号类型（burst / chirp / dsss）× JSR（m10db / m20db / m30db）共 9 组，各自独立跑
  - k_shot=1 表示用 1 条 MeasRes 记录（15 个频段 patch）作为训练集
  - 结果汇总表：行=信号类型，列=JSR 级别

用法:
    python run_rf_open.py
    python run_rf_open.py --k-shot 2 --epochs 50 --gpu-id 1
    python run_rf_open.py --dry-run   # 只打印命令不执行
"""
import argparse
import os
import subprocess

import pandas as pd

from plot_rf_open_heatmap import plot_heatmap as plot_rf_open_heatmap

DATASET      = 'rf_open'
SIGNAL_TYPES = ['burst', 'chirp', 'dsss']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']
K_SHOT       = 1


def read_i_roc(root_dir: str, class_name: str, k_shot: int, seed: int):
    csv_path = os.path.join(
        root_dir, DATASET, class_name, f'k_{k_shot}', 'csv',
        f'Seed_{seed}-results.csv',
    )
    try:
        df = pd.read_csv(csv_path, index_col=0)
        key = f'{DATASET}-{class_name}'
        return round(float(df.loc[key, 'i_roc']), 4)
    except Exception:
        return None


def print_summary_table(root_dir: str, k_shot: int, seed: int):
    col_w = max(len(s) for s in SIGNAL_TYPES) + 2
    header = f'{"Signal":<{col_w}}' + ''.join(f'{nl:>10}' for nl in NOISE_LEVELS)
    sep = '-' * len(header)
    print(f'\n{"="*60}')
    print(f'RF Open Summary  (k-shot={k_shot}, seed={seed})')
    print(f'{"="*60}')
    print(header)
    print(sep)

    col_means = {nl: [] for nl in NOISE_LEVELS}
    for sig in SIGNAL_TYPES:
        row = f'{sig:<{col_w}}'
        for nl in NOISE_LEVELS:
            val = read_i_roc(root_dir, f'{sig}_{nl}', k_shot, seed)
            if val is not None:
                col_means[nl].append(val)
                row += f'{val:>10.4f}'
            else:
                row += f'{"N/A":>10}'
        print(row)

    print(sep)
    mean_row = f'{"Mean":<{col_w}}'
    for nl in NOISE_LEVELS:
        vals = col_means[nl]
        mean_row += f'{sum(vals)/len(vals):>10.4f}' if vals else f'{"N/A":>10}'
    print(mean_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-shot',  type=int,  default=K_SHOT)
    parser.add_argument('--gpu-id',  type=int,  default=0)
    parser.add_argument('--epochs',  type=int,  default=50)
    parser.add_argument('--vis',     type=bool, default=False)
    parser.add_argument('--seed',    type=int,  default=111)
    parser.add_argument('--root-dir', type=str, default='./result')
    parser.add_argument('--dry-run', action='store_true', help='只显示命令不执行')
    args = parser.parse_args()

    vis_str = 'True' if args.vis else 'False'
    run_results = {}

    for sig in SIGNAL_TYPES:
        for nl in NOISE_LEVELS:
            class_name = f'{sig}_{nl}'
            cmd = (
                f'MKL_THREADING_LAYER=GNU python train_cls.py '
                f'--dataset {DATASET} --class_name {class_name} '
                f'--k-shot {args.k_shot} --Epoch {args.epochs} '
                f'--gpu-id {args.gpu_id} --vis {vis_str} --seed {args.seed} '
                f'--root-dir {args.root_dir}'
            )
            print(f'\n{"="*60}')
            print(f'>>> {sig.upper()} | JSR: {nl}')
            print(f'运行: {cmd}')
            if not args.dry_run:
                result = subprocess.run(cmd, shell=True)
                run_results[class_name] = result.returncode == 0
            else:
                run_results[class_name] = None

    if not args.dry_run:
        print_summary_table(args.root_dir, args.k_shot, args.seed)
        out_path = os.path.join(args.root_dir, 'rf_open_heatmap.png')
        plot_rf_open_heatmap(args.root_dir, args.k_shot, args.seed, out_path)
