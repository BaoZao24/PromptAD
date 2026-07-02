"""
频谱异常检测小批量实验：
3 种异常 * 4 个 scene * 3 个 ISR = 36 个实验。

数据切分规则：
- 训练：normal/{ISR} 中前 3/4 的正常图片
- 测试：剩余 1/4 的正常图片 + abnormal/{ISR} 的全部异常图片

用法:
    python run_rf_split_all.py
    python run_rf_split_all.py --gpus 0 1 2 3
    python run_rf_split_all.py --epochs 50 --dry-run
"""
import argparse
import os
import queue
import subprocess
import threading
from itertools import product

import pandas as pd

DATASETS = ['burst_signal', 'chirp_signal', 'dsss_signal', 'pulse_signal', 'wideband_pulse']
SCENES = ['WeaponMuseum_spectrum', 'Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']
DATASET_NOISE_LEVELS = {
    'wideband_pulse': ['m20db', 'm30db', 'm40db'],
    'pulse_signal': ['m20db', 'm30db', 'm40db'],
}
K_SHOT = 1
SPLIT_MODE = 'normal_75_25'


def get_noise_levels(dataset):
    return DATASET_NOISE_LEVELS.get(dataset, NOISE_LEVELS)


def read_i_roc(root_dir, dataset, scene, noise_level, seed):
    candidate_paths = [
        os.path.join(
            root_dir, dataset, scene, noise_level, SPLIT_MODE, f'k_{K_SHOT}', 'csv',
            f'Seed_{seed}-results.csv'
        ),
        os.path.join(
            root_dir, dataset, scene, noise_level, f'k_{K_SHOT}', 'csv',
            f'Seed_{seed}-results.csv'
        ),
    ]
    key = f'{dataset}-{scene}'
    for csv_path in candidate_paths:
        try:
            df = pd.read_csv(csv_path, index_col=0)
            return round(float(df.loc[key, 'i_roc']), 4)
        except Exception:
            continue
    return None


def print_summary_table(root_dir, seed, dataset):
    col_w = max(len(s) for s in SCENES) + 2
    noise_levels = get_noise_levels(dataset)
    header = f'{"Scene":<{col_w}}' + ''.join(f'{nl:>10}' for nl in noise_levels)
    sep = '-' * len(header)
    print(f'\n{"="*60}')
    print(f'Split RF {dataset} Summary  (k-shot={K_SHOT}, split={SPLIT_MODE})')
    print(f'{"="*60}')
    print(header)
    print(sep)

    col_means = {nl: [] for nl in noise_levels}
    for scene in SCENES:
        row = f'{scene:<{col_w}}'
        for nl in noise_levels:
            val = read_i_roc(root_dir, dataset, scene, nl, seed)
            if val is not None:
                col_means[nl].append(val)
                row += f'{val:>10.4f}'
            else:
                row += f'{"N/A":>10}'
        print(row)

    print(sep)
    mean_row = f'{"Mean":<{col_w}}'
    for nl in noise_levels:
        vals = col_means[nl]
        mean_row += f'{sum(vals)/len(vals):>10.4f}' if vals else f'{"N/A":>10}'
    print(mean_row)


_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


def gpu_worker(gpu_id, job_queue, epochs, seed, vis, dry_run, prompt_mode, input_mode, root_dir,
               n_ctx_ab, n_pro_ab, text_prototype_mode,
               cls_score_mode, visual_topk_ratio, visual_score_alpha, visual_score_beta, visual_score_gamma,
               visual_freq_position_weight, normal_dist_ridge,
               visual_adapter, adapter_bottleneck_ratio, adapter_alpha,
               visual_lora, visual_lora_rank, visual_lora_alpha, visual_lora_dropout,
               batch_size, stat_fusion, stat_fusion_beta, stat_topk_ratio,
               multiview_fusion, multiview_fusion_rule, multiview_fusion_lambda):
    while True:
        try:
            dataset, scene, noise_level = job_queue.get_nowait()
        except queue.Empty:
            break

        vis_str = 'True' if vis else 'False'
        cmd = (
            f'MKL_THREADING_LAYER=GNU python train_cls.py '
            f'--dataset {dataset} --class_name {scene} '
            f'--k-shot {K_SHOT} --Epoch {epochs} --gpu-id {gpu_id} '
            f'--noise-level {noise_level} --vis {vis_str} --seed {seed} '
            f'--root-dir {root_dir} '
            f'--prompt-mode {prompt_mode} --input-mode {input_mode} '
            f'--text-prototype-mode {text_prototype_mode} '
            f'--n_ctx_ab {n_ctx_ab} '
            f'--n_pro_ab {n_pro_ab} '
            f'--cls-score-mode {cls_score_mode} '
            f'--visual-topk-ratio {visual_topk_ratio} '
            f'--visual-score-alpha {visual_score_alpha} '
            f'--visual-score-beta {visual_score_beta} '
            f'--visual-score-gamma {visual_score_gamma} '
            f'--visual-freq-position-weight {visual_freq_position_weight} '
            f'--normal-dist-ridge {normal_dist_ridge} '
            f'--visual-adapter {visual_adapter} '
            f'--adapter-bottleneck-ratio {adapter_bottleneck_ratio} '
            f'--adapter-alpha {adapter_alpha} '
            f'--visual-lora {visual_lora} '
            f'--visual-lora-rank {visual_lora_rank} '
            f'--visual-lora-alpha {visual_lora_alpha} '
            f'--visual-lora-dropout {visual_lora_dropout} '
            f'--batch-size {batch_size} '
            f'--stat-fusion {stat_fusion} '
            f'--stat-fusion-beta {stat_fusion_beta} '
            f'--stat-topk-ratio {stat_topk_ratio} '
            f'--multiview-fusion {multiview_fusion} '
            f'--multiview-fusion-rule {multiview_fusion_rule} '
            f'--multiview-fusion-lambda {multiview_fusion_lambda} '
            f'--split-mode {SPLIT_MODE} --normal-train-ratio 0.75'
        )
        safe_print(f'[GPU {gpu_id}] START  {dataset} | {scene} | {noise_level}')
        if not dry_run:
            log_path = f'/tmp/rf_split_{dataset}_{scene}_{noise_level}_gpu{gpu_id}.log'
            with open(log_path, 'w') as flog:
                result = subprocess.run(cmd, shell=True, stdout=flog, stderr=flog)
            rc = result.returncode
            try:
                with open(log_path) as flog:
                    lines = [l.strip() for l in flog if 'AUROC' in l or 'Image-AUROC' in l]
                last = lines[-1] if lines else '(no AUROC in log)'
            except Exception:
                last = '(log read error)'
            status = 'OK' if rc == 0 else f'FAILED(rc={rc})'
            safe_print(f'[GPU {gpu_id}] {status} {dataset} | {scene} | {noise_level} | {last}')
        else:
            safe_print(f'[GPU {gpu_id}] DRY-RUN: {cmd}')

        job_queue.task_done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='使用的 GPU 列表，默认自动检测全部')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--root-dir', type=str, default='./result')
    parser.add_argument('--prompt-mode', type=str, default='rf',
                        choices=['generic', 'rf_domain', 'rf', 'legacy', 'rf_object_agnostic',
                                 'rf_scene_conditioned', 'rf_signal_structured'])
    parser.add_argument('--n-ctx-ab', type=int, default=1,
                        help='learned abnormal prompt 中每组 abnormal_ctx 的 token 数')
    parser.add_argument('--n-pro-ab', type=int, default=4,
                        help='learned abnormal prompt 的组数')
    parser.add_argument('--text-prototype-mode', type=str, default='single',
                        choices=['single', 'grouped_max', 'grouped_mean', 'grouped_meanmax', 'grouped_softmax'],
                        help='single 平均所有异常 prompt；grouped_max 保留 burst/chirp/dsss 多异常原型并取最大异常分数；grouped_meanmax 使用 0.5*mean + 0.5*max 融合多异常原型分数')
    parser.add_argument('--cls-score-mode', type=str, default='text_only',
                        choices=['text_only', 'visual_topk', 'visual_topk_max', 'visual_topk_freq',
                                 'normal_center', 'normal_mahalanobis', 'text_normal_center', 'text_normal_mahalanobis'],
                        help='图像级分数融合方式')
    parser.add_argument('--visual-topk-ratio', type=float, default=0.05,
                        help='visual patch score 聚合时使用的 top-k 比例')
    parser.add_argument('--visual-score-alpha', type=float, default=1.0,
                        help='textual score 融合权重')
    parser.add_argument('--visual-score-beta', type=float, default=1.0,
                        help='visual top-k score 融合权重')
    parser.add_argument('--visual-score-gamma', type=float, default=0.0,
                        help='visual max score 融合权重')
    parser.add_argument('--visual-freq-position-weight', type=float, default=0.0,
                        help='频率位置约束强度，用于强调远离中心或特定频带的异常')
    parser.add_argument('--normal-dist-ridge', type=float, default=1e-4,
                        help='normal distribution diagonal variance 的最小平滑项')
    parser.add_argument('--input-mode', type=str, default='auto',
                        choices=['auto', 'rgb', 'gray3', 'gray_local2d_edge', 'morph_fusion_gray_residual_a01', 'morph_fusion_local2d_residual_a01', 'morph_fusion_tophat_a01', 'morph_fusion_multiscale_residual_a01', 'morph_fusion_gray_residual_no_contrast_a01', 'morph_fusion_clahe_gray'])
    parser.add_argument('--visual-adapter', action='store_true', default=False,
                        help='启用冻结 CLIP 后的轻量残差 visual adapter')
    parser.add_argument('--adapter-bottleneck-ratio', type=float, default=0.25)
    parser.add_argument('--adapter-alpha', type=float, default=0.2)
    parser.add_argument('--visual-lora', action='store_true', default=False,
                        help='启用 CLIP visual transformer attention LoRA')
    parser.add_argument('--visual-lora-rank', type=int, default=4)
    parser.add_argument('--visual-lora-alpha', type=float, default=8.0)
    parser.add_argument('--visual-lora-dropout', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--stat-fusion', action='store_true', default=False,
                        help='启用传统频谱统计分数直接融合，不做 z-score')
    parser.add_argument('--stat-fusion-beta', type=float, default=0.5)
    parser.add_argument('--stat-topk-ratio', type=float, default=0.05)
    parser.add_argument('--multiview-fusion', action='store_true', default=False,
                        help='启用固定三视图在线分数融合')
    parser.add_argument('--multiview-fusion-rule', type=str, default='mean',
                        choices=['max', 'mean', 'conservative_lam0.5'])
    parser.add_argument('--multiview-fusion-lambda', type=float, default=0.5)
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS, choices=DATASETS,
                        help='要运行的数据集子集，默认运行全部')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.gpus is None:
        try:
            result = subprocess.run(
                'nvidia-smi --query-gpu=index --format=csv,noheader',
                shell=True, capture_output=True, text=True)
            args.gpus = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        except Exception:
            args.gpus = [0]

    print(f'使用 GPU: {args.gpus}')
    print(f'数据集: {args.datasets}  |  Scenes: {SCENES}')
    print(f'训练切分: split={SPLIT_MODE}, normal_ratio=0.75, k-shot={K_SHOT}, epochs={args.epochs}')

    job_q = queue.Queue()
    all_jobs = [
        (dataset, scene, noise)
        for dataset in args.datasets
        for scene in SCENES
        for noise in get_noise_levels(dataset)
    ]
    skipped, pending = [], []
    for dataset, scene, noise_level in all_jobs:
        val = read_i_roc(args.root_dir, dataset, scene, noise_level, args.seed)
        if not args.force and val is not None and val > 0:
            skipped.append((dataset, scene, noise_level))
        else:
            pending.append((dataset, scene, noise_level))
            job_q.put((dataset, scene, noise_level))

    if skipped:
        print(f'跳过已完成 {len(skipped)} 个 job:')
        for j in skipped:
            print(f'  ✓ {j[0]} | {j[1]} | {j[2]}')
    print(f'待运行 {len(pending)} 个 job，分配到 {len(args.gpus)} 个 GPU\n')

    threads = []
    for gid in args.gpus:
        t = threading.Thread(
            target=gpu_worker,
            args=(gid, job_q, args.epochs, args.seed, args.vis, args.dry_run,
                  args.prompt_mode, args.input_mode, args.root_dir,
                  args.n_ctx_ab, args.n_pro_ab, args.text_prototype_mode,
                  args.cls_score_mode, args.visual_topk_ratio, args.visual_score_alpha, args.visual_score_beta, args.visual_score_gamma,
                  args.visual_freq_position_weight, args.normal_dist_ridge,
                  args.visual_adapter, args.adapter_bottleneck_ratio, args.adapter_alpha,
                  args.visual_lora, args.visual_lora_rank, args.visual_lora_alpha, args.visual_lora_dropout,
                  args.batch_size, args.stat_fusion, args.stat_fusion_beta, args.stat_topk_ratio,
                  args.multiview_fusion, args.multiview_fusion_rule, args.multiview_fusion_lambda),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if not args.dry_run:
        for dataset in args.datasets:
            print_summary_table(args.root_dir, args.seed, dataset)
