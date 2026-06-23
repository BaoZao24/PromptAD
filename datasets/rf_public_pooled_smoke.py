"""池化版 smoke source loader for RF cross-public training.

按 docs/PromptAD_adapter_experiment_plan.md 联合训练的最新设计:
    把 burst/chirp/dsss 三类 source 数据池化到一个 dataset, 训一个共享 prompt
    (class_name='signal') 的 PromptAD 模型. target 端仍保持三类分别评估.

samples per class (300 normal + 300 abnormal) × 3 classes = 1800 张池化训练样本:
    - normal 来自 RF_SPE_PNG/RF_Spectrum_Public_Dataset (跨 class 共享),
      用 round-robin 把 normal 池子三等分, 保证三类无重复
    - abnormal 来自 RF_SPE_PNG/{class}/abnormal/{noise_level}/
      noise level per-class:
        burst m10db (442 张)
        chirp m30db (544 张)  <- chirp 只有 m30db
        dsss  m10db (544 张)

      本地 chirp 的 m10db 是空, 只有 m30db. 先记录该异质性, 不强行对齐.

train_data 仅保留 16 张 source normal 用于 build_image_feature_gallery, 剩余的
都进入 test_data 充当 supervised query (loss 用的就是 phase='test' 那批).
"""

import os
from pathlib import Path

from .rf_spe_png import (
    RF_PUBLIC_NORMAL_DIR,
    RF_SPE_PNG_DIR,
    _filter_by_freq,
    _list_public_record_dirs,
)


rf_public_pooled_smoke_classes = ['signal']


POOL_SIGNAL_TYPES = ['burst', 'chirp', 'dsss']

# 默认每 class 的 normal / abnormal 上限. 经数据盘点, chirp 的 m10db 没有 abnormal,
# 所以 chirp 的 abnormal 退到 m30db; 文字记录在 README/SUMMARY 里.
DEFAULT_PER_CLASS_NORMAL = 300
DEFAULT_PER_CLASS_ABNORMAL = 300

ABNORMAL_NOISE_FALLBACK = {
    'burst': ['m10db', 'm20db', 'm30db'],
    'chirp': ['m30db', 'm20db', 'm10db'],   # chirp 只有 m30db 有 abnormal
    'dsss':  ['m10db', 'm20db', 'm30db'],
}


def _resolve_abnormal_dir(sig: str, requested_noise: str) -> tuple:
    """优先用请求的 noise level, 否则按 fallback 顺序找第一个有内容的."""
    candidates = [requested_noise] + [n for n in ABNORMAL_NOISE_FALLBACK[sig] if n != requested_noise]
    for nl in candidates:
        abn_dir = os.path.join(RF_SPE_PNG_DIR, sig, 'abnormal', nl)
        gt_dir = os.path.join(RF_SPE_PNG_DIR, sig, 'groundtruth', nl)
        if os.path.isdir(abn_dir):
            files = list(Path(abn_dir).glob('*.png'))
            if files:
                return abn_dir, gt_dir, nl
    return None, None, None


def _round_robin_partition(items, n_parts):
    """把 items 按 round-robin 分到 n_parts 个 disjoint 子集, 保证三类 normal 无重复."""
    parts = [[] for _ in range(n_parts)]
    for idx, item in enumerate(items):
        parts[idx % n_parts].append(item)
    return parts


def load_rf_public_pooled_smoke(
    category='signal',
    k_shot=1,
    noise_level='m10db',
    freq=None,
    train_category=None,
    split_mode='normal_75_25',
    normal_train_ratio=0.75,
    max_normal_per_class=DEFAULT_PER_CLASS_NORMAL,
    max_abnormal_per_class=DEFAULT_PER_CLASS_ABNORMAL,
    pool_classes=None,
):
    if category != 'signal':
        raise ValueError(
            f"rf_public_pooled_smoke 只支持 category='signal' (共享 prompt), got {category!r}"
        )

    pool_classes = list(pool_classes) if pool_classes else list(POOL_SIGNAL_TYPES)
    invalid = [c for c in pool_classes if c not in POOL_SIGNAL_TYPES]
    if invalid:
        raise ValueError(f'pool_classes 含非法类型 {invalid}, 仅允许 {POOL_SIGNAL_TYPES}')

    record_dirs = _list_public_record_dirs()
    selected_train_dirs = record_dirs[:k_shot]
    selected_query_dirs = record_dirs[k_shot:]

    # gallery: 取 k_shot 个 record dir 全部 normal (~16 张). 不分 class.
    train_normals = []
    for d in selected_train_dirs:
        train_normals.extend(sorted(str(p) for p in d.glob('*.png')))
    train_normals = _filter_by_freq(train_normals, freq)

    # query 用的 normal 池: 把剩余 record dir 的 normal 全部展开
    query_normal_pool = []
    for d in selected_query_dirs:
        query_normal_pool.extend(sorted(str(p) for p in d.glob('*.png')))
    query_normal_pool = _filter_by_freq(query_normal_pool, freq)

    # round-robin 切成 len(pool_classes) 份, 每份取前 max_normal_per_class 张
    rr_parts = _round_robin_partition(query_normal_pool, len(pool_classes))

    # ---- gallery (train_data) ----
    train_imgs = list(train_normals)
    train_gts = [0] * len(train_imgs)
    train_labels = [0] * len(train_imgs)
    train_types = ['pooled_train_normal' for _ in train_imgs]

    # ---- query (test_data) ----
    test_imgs, test_gts, test_labels, test_types = [], [], [], []

    for sig_idx, sig in enumerate(pool_classes):
        # normals for this signal slot
        sig_normals = rr_parts[sig_idx][:max_normal_per_class] if max_normal_per_class else rr_parts[sig_idx]
        for p in sig_normals:
            test_imgs.append(p)
            test_gts.append(0)
            test_labels.append(0)
            test_types.append(f'pooled_{sig}_normal')

        # abnormals for this signal type
        abn_dir, gt_dir, used_nl = _resolve_abnormal_dir(sig, noise_level)
        if abn_dir is None:
            print(f'[rf_public_pooled_smoke] WARNING: no abnormal samples found for {sig}; skipping abnormals')
            continue
        if used_nl != noise_level:
            print(f'[rf_public_pooled_smoke] note: {sig} abnormal falls back to {used_nl} (no {noise_level} on disk)')
        abn_paths = sorted(str(p) for p in Path(abn_dir).glob('*.png'))
        abn_paths = _filter_by_freq(abn_paths, freq)
        if max_abnormal_per_class:
            abn_paths = abn_paths[:max_abnormal_per_class]

        for img_path in abn_paths:
            img_name = os.path.basename(img_path).replace('_abnormal.png', '_groundtruth.png')
            gt_path = os.path.join(gt_dir, img_name)
            test_imgs.append(img_path)
            test_gts.append(gt_path if os.path.exists(gt_path) else 0)
            test_labels.append(1)
            test_types.append(f'pooled_{sig}_abnormal_{used_nl}')

    return (train_imgs, train_gts, train_labels, train_types), \
           (test_imgs, test_gts, test_labels, test_types)
