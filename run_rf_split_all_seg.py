#!/usr/bin/env python
"""Dispatch PromptAD seg experiments (train_seg.py) across RF jobs on 4 GPUs.

Default protocol matches the original seg baseline:
  prompt_mode=legacy, input_mode=rgb, normal_75_25, k_shot=1, seed=111,
  epochs=50, batch_size=400, vis=False.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

DATASETS_ALL = ['burst_signal', 'chirp_signal', 'dsss_signal', 'pulse_signal']
SCENES = [
    'WeaponMuseum_spectrum',
    'Playground_spectrum',
    'TimeSquare_spectrum',
    'Gymnasium_spectrum',
]
JSR_BY_DATASET = {
    'burst_signal': ['m10db', 'm20db', 'm30db'],
    'chirp_signal': ['m10db', 'm20db', 'm30db'],
    'dsss_signal':  ['m10db', 'm20db', 'm30db'],
    'pulse_signal': ['m20db', 'm30db', 'm40db'],
}


def build_cmd(dataset: str, scene: str, jsr: str, gpu: int, args) -> list[str]:
    cmd = [
        'python', '-u', 'train_seg.py',
        '--dataset', dataset,
        '--class_name', scene,
        '--k-shot', '1',
        '--Epoch', str(args.epochs),
        '--gpu-id', str(gpu),
        '--noise-level', jsr,
        '--vis', 'False',
        '--seed', str(args.seed),
        '--root-dir', args.root_dir,
        '--prompt-mode', args.prompt_mode,
        '--input-mode', args.input_mode,
        '--batch-size', str(args.batch_size),
        '--eval-every', str(args.eval_every),
        '--split-mode', 'normal_75_25',
        '--normal-train-ratio', '0.75',
    ]
    if args.dense_mask_branch:
        cmd.extend([
            '--dense-mask-branch', 'True',
            '--dense-mask-hidden-ratio', str(args.dense_mask_hidden_ratio),
            '--dense-mask-loss-weight', str(args.dense_mask_loss_weight),
        ])
        if args.dense_mask_lr is not None:
            cmd.extend(['--dense-mask-lr', str(args.dense_mask_lr)])
    if args.visual_class_prompt:
        cmd.extend([
            '--visual-class-prompt', 'True',
            '--visual-class-token-num', str(args.visual_class_token_num),
            '--visual-class-prompt-bottleneck-ratio', str(args.visual_class_prompt_bottleneck_ratio),
            '--visual-class-prompt-alpha', str(args.visual_class_prompt_alpha),
            '--visual-class-prototype-mode', args.visual_class_prototype_mode,
            '--visual-class-prototype-num', str(args.visual_class_prototype_num),
        ])
        if args.visual_class_prompt_lr is not None:
            cmd.extend(['--visual-class-prompt-lr', str(args.visual_class_prompt_lr)])
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--datasets', nargs='+', type=str, default=DATASETS_ALL,
                   choices=DATASETS_ALL,
                   help='Which anomaly types to run; defaults to all four.')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=111)
    p.add_argument('--root-dir', type=str, required=True)
    p.add_argument('--prompt-mode', type=str, default='legacy')
    p.add_argument('--input-mode', type=str, default='rgb')
    p.add_argument('--batch-size', type=int, default=400)
    p.add_argument('--eval-every', type=int, default=1,
                   help='seg 每隔多少个 epoch 做一次全量评估；默认每轮都评估')
    p.add_argument('--dense-mask-branch', action='store_true')
    p.add_argument('--dense-mask-hidden-ratio', type=float, default=0.25)
    p.add_argument('--dense-mask-loss-weight', type=float, default=1.0)
    p.add_argument('--dense-mask-lr', type=float, default=None)
    p.add_argument('--visual-class-prompt', action='store_true',
                   help='启用 VCPA（visual class prompt adapter）')
    p.add_argument('--visual-class-token-num', type=int, default=2)
    p.add_argument('--visual-class-prompt-bottleneck-ratio', type=float, default=0.25)
    p.add_argument('--visual-class-prompt-alpha', type=float, default=0.2)
    p.add_argument('--visual-class-prototype-mode', type=str, default='mean',
                   choices=['mean', 'diverse'])
    p.add_argument('--visual-class-prototype-num', type=int, default=1)
    p.add_argument('--visual-class-prompt-lr', type=float, default=None)
    p.add_argument('--log-dir', type=str, required=True,
                   help='Per-job stdout/stderr will be written under here.')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='Re-run even if result CSV already exists.')
    args = p.parse_args()

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    for dataset in args.datasets:
        for scene in SCENES:
            for jsr in JSR_BY_DATASET[dataset]:
                jobs.append((dataset, scene, jsr))

    # Filter out jobs whose result CSV already exists, unless --force
    if not args.force:
        kept = []
        for dataset, scene, jsr in jobs:
            base = Path(args.root_dir) / dataset / scene / jsr
            candidates = [
                base / 'normal_75_25' / 'k_1' / 'csv' / f'Seed_{args.seed}-results.csv',
                base / 'k_1' / 'csv' / f'Seed_{args.seed}-results.csv',
            ]
            if any(c.exists() for c in candidates):
                print(f'[SKIP] {dataset}|{scene}|{jsr} (csv exists)')
                continue
            kept.append((dataset, scene, jsr))
        jobs = kept

    print(f'Dispatching {len(jobs)} jobs across {len(args.gpus)} GPUs '
          f'(prompt={args.prompt_mode}, input={args.input_mode}, '
          f'epochs={args.epochs}, eval_every={args.eval_every}, seed={args.seed}, '
          f'dense={args.dense_mask_branch}, vcpa={args.visual_class_prompt})')

    if args.dry_run:
        for j in jobs:
            print('DRY-RUN', j, 'cmd:', ' '.join(build_cmd(*j, gpu=args.gpus[0], args=args)))
        return

    job_q: queue.Queue = queue.Queue()
    for j in jobs:
        job_q.put(j)

    results = []
    results_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                dataset, scene, jsr = job_q.get_nowait()
            except queue.Empty:
                return
            tag = f'{dataset}|{scene}|{jsr}'
            log_file = log_dir / f'{dataset}__{scene}__{jsr}.log'
            print(f'[GPU {gpu}] START  {tag}', flush=True)
            t0 = time.time()
            cmd = build_cmd(dataset, scene, jsr, gpu, args)
            env = os.environ.copy()
            env['MKL_THREADING_LAYER'] = 'GNU'
            with log_file.open('w') as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            rc = proc.returncode
            dt = time.time() - t0
            status = 'OK ' if rc == 0 else f'FAIL({rc})'
            print(f'[GPU {gpu}] {status} {tag} dt={dt:.1f}s', flush=True)
            with results_lock:
                results.append({'gpu': gpu, 'job': tag, 'rc': rc, 'dt_s': dt})
            job_q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    fails = [r for r in results if r['rc'] != 0]
    print('\n========== SUMMARY ==========')
    print(f'total jobs run: {len(results)}; failures: {len(fails)}')
    for r in fails:
        print(' FAIL:', r['job'], 'rc=', r['rc'])
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
