"""
跨站点测试：用 WeaponMuseum_spectrum 的正常样本训练（k-shot=24），对所有 JSR 级别的其他站点做 cross-site 测试，
运行结束后汇总输出结果表。

用法:
    python run_cross_site_dsss.py
    python run_cross_site_dsss.py --epochs 100 --gpu-id 1
    python run_cross_site_dsss.py --dry-run   # 只显示命令不执行
"""
import argparse
import os
import subprocess

import pandas as pd

DATASET = 'dsss_signal'
TRAIN_SITE = 'WeaponMuseum_spectrum'
TEST_SCENES = ['Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']
K_SHOT = 24


def read_i_roc(root_dir, dataset, train_site, test_scene, noise_level, seed):
    dir_class = f'{train_site}_to_{test_scene}'
    csv_path = os.path.join(root_dir, dataset, dir_class, noise_level,
                            f'k_{K_SHOT}', 'csv', f'Seed_{seed}-results.csv')
    try:
        df = pd.read_csv(csv_path, index_col=0)
        key = f'{dataset}-{test_scene}'
        return round(float(df.loc[key, 'i_roc']), 4)
    except Exception:
        return None


def print_summary_table(root_dir, seed):
    col_w = max(len(s) for s in TEST_SCENES) + 2
    header = f'{"Scene":<{col_w}}' + ''.join(f'{nl:>10}' for nl in NOISE_LEVELS)
    sep = '-' * len(header)
    print(f'\n{"="*60}')
    print(f'Cross-site {DATASET} Summary  (Train: {TRAIN_SITE}, k-shot={K_SHOT})')
    print(f'{"="*60}')
    print(header)
    print(sep)

    col_means = {nl: [] for nl in NOISE_LEVELS}
    for scene in TEST_SCENES:
        row = f'{scene:<{col_w}}'
        for nl in NOISE_LEVELS:
            val = read_i_roc(root_dir, DATASET, TRAIN_SITE, scene, nl, seed)
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
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--root-dir', type=str, default='./result')
    parser.add_argument('--prompt-mode', type=str, default='rf',
                        choices=['rf', 'legacy', 'rf_object_agnostic', 'rf_scene_conditioned'])
    parser.add_argument('--input-mode', type=str, default='auto',
                        choices=['auto', 'rgb', 'spectral_gradient', 'signal_adaptive'])
    parser.add_argument('--dry-run', action='store_true', help='只显示命令不执行')
    args = parser.parse_args()

    vis_str = 'True' if args.vis else 'False'
    run_results = {}

    for noise_level in NOISE_LEVELS:
        print(f'\n{"="*60}')
        print(f'>>> Cross-site {DATASET} | Train: {TRAIN_SITE} | JSR: {noise_level}')
        print(f'{"="*60}')
        for scene in TEST_SCENES:
            cmd = (
                f'MKL_THREADING_LAYER=GNU python train_cls.py --dataset {DATASET} --class_name {scene} '
                f'--train-site {TRAIN_SITE} '
                f'--k-shot {K_SHOT} --Epoch {args.epochs} --gpu-id {args.gpu_id} '
                f'--noise-level {noise_level} --vis {vis_str} --seed {args.seed} '
                f'--prompt-mode {args.prompt_mode} --input-mode {args.input_mode}'
            )
            print(f'\n--- {TRAIN_SITE} -> {scene} ({noise_level}) ---')
            print(f'运行: {cmd}')
            if not args.dry_run:
                result = subprocess.run(cmd, shell=True)
                run_results[(scene, noise_level)] = result.returncode == 0
            else:
                run_results[(scene, noise_level)] = None

    if not args.dry_run:
        print_summary_table(args.root_dir, args.seed)
