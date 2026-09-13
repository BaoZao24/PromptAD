"""Reproduce the current In-house configuration without using physical GPU0."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis_outputs/20260911_inhouse_1nn_no_tta'
PY = '/home/wangbei/miniconda3/envs/prompt_ad/bin/python'
MANIFEST = 'analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json'


def run(name, argv):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT)
    with (OUT / (name + '.log')).open('w') as log:
        print('START', name, flush=True)
        subprocess.run([PY, *argv], cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
        print('DONE', name, flush=True)


def branch(kind):
    folder = str(OUT / kind)
    if kind == 'vit':
        base = ['tools/eval_cls_vit_patchcore_gallery.py', '--gpu-id', '1',
                '--nn-topk', '1', '--paired-tta', 'none', '--support-augment', 'none',
                '--coreset-ratio', '0.5', '--coreset-method', 'farthest',
                '--gallery-chunk-size', '1024']
    else:
        base = ['tools/eval_cls_aux_cnn_gallery.py', '--gpu-id', '2',
                '--protocol', 'rf_target', '--cnn-encoder', 'resnet18',
                '--cnn-feature-mode', 'layer3', '--max-gallery-patches', '0',
                '--cnn-coreset-ratio', '0.5', '--distance-chunk-size', '512']
    base += ['--support-manifest', MANIFEST, '--normal-sampling', 'per_frequency',
             '--batch-size', '16', '--num-workers', '2', '--seed', '111']
    run(kind + '_reference', base + ['--output-root', folder + '_reference', '--support-reference-only'])
    run(kind, base + ['--output-root', folder])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'run_protocol.json').write_text(json.dumps(dict(
        dataset='In-house RF', nearest_neighbors=1, tta='none',
        support_augmentation='none', vit_coreset_ratio=.5, cnn_coreset_ratio=.5,
        manifest=MANIFEST, seed=111, physical_gpus=[1, 2],
        batch_size=16, workers_per_branch=2,
        checkpoint=None, encoder_weights='frozen pretrained visual encoders',
        calibration='original normal features; exclude exact self-match'), indent=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(branch, kind) for kind in ('vit', 'cnn')]
        for future in futures:
            future.result()
    run('fusion', ['tools/eval_cls_dual_visual_evidence_fusion.py',
                  '--protocol', 'rf_target', '--gate-protocol', 'support_only',
                  '--vit-score-dir', str(OUT / 'vit/scores'),
                  '--cnn-score-dir', str(OUT / 'cnn/scores'),
                  '--vit-reference-dir', str(OUT / 'vit_reference'),
                  '--cnn-reference-dir', str(OUT / 'cnn_reference'),
                  '--support-manifest', MANIFEST,
                  '--output-root', str(OUT / 'fusion')])
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
