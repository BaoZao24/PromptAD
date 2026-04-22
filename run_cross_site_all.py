"""
并行跨站点测试：用 BinBo 正常样本训练（k-shot=24），
对所有信号类型 × 所有 JSR 级别 × 其他站点做 cross-site 测试。
将所有 job 分发到可用 GPU 上并行运行，完成后打印汇总表。

用法:
    python run_cross_site_all.py                     # 自动使用所有 GPU
    python run_cross_site_all.py --gpus 0 1 2 3      # 指定 GPU
    python run_cross_site_all.py --epochs 100
    python run_cross_site_all.py --dry-run
"""
import argparse
import os
import queue
import subprocess
import threading
from itertools import product

import pandas as pd

TRAIN_SITE = 'WeaponMuseum_spectrum'
TEST_SCENES = ['Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']
DATASETS = ['dsss_signal', 'burst_signal', 'chirp_signal']
K_SHOT = 24


# --------------------------------------------------------------------------- #
#  结果读取 & 汇总表
# --------------------------------------------------------------------------- #

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


def print_summary_table(root_dir, seed, dataset):
    col_w = max(len(s) for s in TEST_SCENES) + 2
    header = f'{"Scene":<{col_w}}' + ''.join(f'{nl:>10}' for nl in NOISE_LEVELS)
    sep = '-' * len(header)
    print(f'\n{"="*60}')
    print(f'Cross-site {dataset} Summary  (Train: {TRAIN_SITE}, k-shot={K_SHOT})')
    print(f'{"="*60}')
    print(header)
    print(sep)

    col_means = {nl: [] for nl in NOISE_LEVELS}
    for scene in TEST_SCENES:
        row = f'{scene:<{col_w}}'
        for nl in NOISE_LEVELS:
            val = read_i_roc(root_dir, dataset, TRAIN_SITE, scene, nl, seed)
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


# --------------------------------------------------------------------------- #
#  并行 GPU 调度
# --------------------------------------------------------------------------- #

_print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


def gpu_worker(gpu_id, job_queue, epochs, seed, vis, dry_run):
    """每个 GPU 独占一个 worker 线程，顺序消费 job_queue 里的任务"""
    while True:
        try:
            dataset, scene, noise_level = job_queue.get_nowait()
        except queue.Empty:
            break

        vis_str = 'True' if vis else 'False'
        cmd = (
            f'MKL_THREADING_LAYER=GNU python train_cls.py '
            f'--dataset {dataset} --class_name {scene} '
            f'--train-site {TRAIN_SITE} '
            f'--k-shot {K_SHOT} --Epoch {epochs} --gpu-id {gpu_id} '
            f'--noise-level {noise_level} --vis {vis_str} --seed {seed}'
        )
        safe_print(f'[GPU {gpu_id}] START  {dataset} | {TRAIN_SITE}->{scene} | {noise_level}')
        if not dry_run:
            log_path = f'/tmp/cross_{dataset}_{scene}_{noise_level}_gpu{gpu_id}.log'
            with open(log_path, 'w') as flog:
                result = subprocess.run(cmd, shell=True, stdout=flog, stderr=flog)
            rc = result.returncode
            # 从日志尾部取最后一行带 AUROC 的输出
            try:
                with open(log_path) as flog:
                    lines = [l.strip() for l in flog if 'AUROC' in l or 'Image-AUROC' in l]
                last = lines[-1] if lines else '(no AUROC in log)'
            except Exception:
                last = '(log read error)'
            status = 'OK' if rc == 0 else f'FAILED(rc={rc})'
            safe_print(f'[GPU {gpu_id}] {status} {dataset} | {TRAIN_SITE}->{scene} | {noise_level} | {last}')
        else:
            safe_print(f'[GPU {gpu_id}] DRY-RUN: {cmd}')

        job_queue.task_done()


# --------------------------------------------------------------------------- #
#  主入口
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='使用的 GPU 列表，默认自动检测全部')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--root-dir', type=str, default='./result')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='保存 scoremap 可视化图像')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制重跑已完成的 job')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # 自动检测 GPU 数量
    if args.gpus is None:
        try:
            result = subprocess.run(
                'nvidia-smi --query-gpu=index --format=csv,noheader',
                shell=True, capture_output=True, text=True)
            args.gpus = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        except Exception:
            args.gpus = [0]

    print(f'使用 GPU: {args.gpus}')
    print(f'训练站点: {TRAIN_SITE}  |  测试站点: {TEST_SCENES}')
    print(f'数据集: {DATASETS}  |  JSR: {NOISE_LEVELS}  |  k-shot={K_SHOT}  |  epochs={args.epochs}')

    # 构建全部 job，跳过已有有效结果的
    job_q = queue.Queue()
    all_jobs = list(product(DATASETS, TEST_SCENES, NOISE_LEVELS))
    skipped, pending = [], []
    for dataset, scene, noise_level in all_jobs:
        val = read_i_roc(args.root_dir, dataset, TRAIN_SITE, scene, noise_level, args.seed)
        if not args.force and val is not None and val > 0:
            skipped.append((dataset, scene, noise_level))
        else:
            pending.append((dataset, scene, noise_level))
            job_q.put((dataset, scene, noise_level))

    if skipped:
        print(f'跳过已完成 {len(skipped)} 个 job:')
        for j in skipped:
            print(f'  ✓ {j[0]} | {TRAIN_SITE}->{j[1]} | {j[2]}')
    print(f'待运行 {len(pending)} 个 job，分配到 {len(args.gpus)} 个 GPU\n')

    # 启动 worker 线程（每 GPU 一个）
    threads = []
    for gid in args.gpus:
        t = threading.Thread(
            target=gpu_worker,
            args=(gid, job_q, args.epochs, args.seed, args.vis, args.dry_run),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # 汇总结果表
    if not args.dry_run:
        for dataset in DATASETS:
            print_summary_table(args.root_dir, args.seed, dataset)
        # 绘制热力图
        import subprocess as sp
        sp.run(f'python plot_heatmap.py --root-dir {args.root_dir} --seed {args.seed}', shell=True)
