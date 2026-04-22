"""
批量运行 chirp_signal 实验（所有场景 + 指定噪声级别），包含可视化

用法:
    python run_chirp.py --noise-level m30db --k-shot 4
    python run_chirp.py --noise-level m30db --k-shot 4 --epochs 100
    python run_chirp.py --all-jsr --k-shot 4  # 运行所有 JSR 级别
"""
import argparse
import subprocess
import os

SCENES = ['WeaponMuseum_spectrum', 'Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-level', type=str, default='m30db', choices=['m10db', 'm20db', 'm30db'])
    parser.add_argument('--all-jsr', action='store_true', help='运行所有 JSR 级别 (m10db, m20db, m30db)')
    parser.add_argument('--k-shot', type=int, default=24)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--vis', type=bool, default=True, help='Enable visualization')
    parser.add_argument('--dry-run', action='store_true', help='只显示命令不执行')
    args = parser.parse_args()

    vis_str = 'True' if args.vis else 'False'

    noise_levels = ['m10db', 'm20db', 'm30db'] if args.all_jsr else [args.noise_level]

    for noise_level in noise_levels:
        print(f'\n{"="*60}')
        print(f'>>> JSR level: {noise_level}')
        print(f'{"="*60}')
        results = {}
        for scene in SCENES:
            cmd = (
                f'python train_cls.py --dataset chirp_signal --class_name {scene} '
                f'--k-shot {args.k_shot} --Epoch {args.epochs} --gpu-id {args.gpu_id} '
                f'--noise-level {noise_level} --vis {vis_str}'
            )
            print(f'\n--- {scene} ({noise_level}) ---')
            print(f'运行: {cmd}')
            if not args.dry_run:
                result = subprocess.run(cmd, shell=True)
                results[scene] = 'done' if result.returncode == 0 else f'failed (code={result.returncode})'
            else:
                results[scene] = 'dry-run'

        print(f'\n--- {noise_level} 结果汇总 ---')
        for scene, status in results.items():
            print(f'  {scene}: {status}')
