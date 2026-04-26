"""
RF Open 公开数据集 loader

数据结构:
/mnt/data/wangbei/data/datasets/
├── train/
│   └── normal/                          # 2295 张正常训练 patch（153 条测量记录 × 15 个频段）
└── test/
    ├── normal/                          # 2295 张正常测试 patch
    ├── burst/{m10db,m20db,m30db}/       # burst 异常 patch（仅 _abnormal，无 GT mask）
    ├── chirp/{m10db,m20db,m30db}/       # chirp 异常 patch
    └── dsss/{m10db,m20db,m30db}/        # DSSS 异常 patch

class name 格式：'{signal_type}_{noise_level}'，例如 'burst_m10db'
k_shot：MeasRes 测量记录数，每条记录包含 15 个频段 patch，
        k_shot=1 → 取第 1 条记录的 15 张 patch 作为训练集。
"""
import os
import re
from collections import defaultdict
from pathlib import Path


RF_OPEN_TRAIN_DIR = '/mnt/data/wangbei/data/datasets/train/normal'
RF_OPEN_TEST_DIR  = '/mnt/data/wangbei/data/datasets/test'

SIGNAL_TYPES  = ['burst', 'chirp', 'dsss']
NOISE_LEVELS  = ['m10db', 'm20db', 'm30db']

rf_open_classes = [
    f'{sig}_{jsr}'
    for sig in SIGNAL_TYPES
    for jsr in NOISE_LEVELS
]


def _extract_record_id(filename: str) -> str:
    """从文件名提取 MeasRes 记录 ID（用于 k-shot 按记录采样）。
    例: MeasRes_0770_0032_f94.10-96.54MHz_patch006.png -> '0770_0032'
    """
    m = re.search(r'MeasRes_(\d+_\d+)', filename)
    return m.group(1) if m else ''


def load_rf_open(category: str, k_shot: int):
    """
    Args:
        category: '{signal_type}_{noise_level}'，例如 'burst_m10db'
        k_shot:   训练用的 MeasRes 记录数。每条记录有 15 个频段 patch，
                  实际训练图片数 = k_shot × 15。

    Returns:
        train_data: (img_paths, gt_paths, labels, types)
        test_data:  (img_paths, gt_paths, labels, types)
    """
    signal_type, noise_level = category.rsplit('_', 1)

    # ------------------------------------------------------------------ train
    # 按记录 ID 分组，取前 k_shot 条记录的全部 patch
    record_groups: dict = defaultdict(list)
    for img_path in sorted(Path(RF_OPEN_TRAIN_DIR).glob('*.png')):
        rec_id = _extract_record_id(img_path.name)
        if rec_id:
            record_groups[rec_id].append(str(img_path))

    selected_records = sorted(record_groups.keys())[:k_shot]

    train_img_paths, train_labels, train_types = [], [], []
    for rec_id in selected_records:
        for p in sorted(record_groups[rec_id]):
            train_img_paths.append(p)
            train_labels.append(0)
            train_types.append(f'{category}_train_normal')

    train_gt_paths = [0] * len(train_img_paths)

    # ------------------------------------------------------------------ test
    test_img_paths, test_gt_paths, test_labels, test_types = [], [], [], []

    # 正常测试样本
    test_normal_dir = os.path.join(RF_OPEN_TEST_DIR, 'normal')
    if os.path.isdir(test_normal_dir):
        for img_path in sorted(Path(test_normal_dir).glob('*.png')):
            test_img_paths.append(str(img_path))
            test_gt_paths.append(0)
            test_labels.append(0)
            test_types.append(f'{category}_normal')

    # 异常测试样本
    test_abnormal_dir = os.path.join(RF_OPEN_TEST_DIR, signal_type, noise_level)
    if os.path.isdir(test_abnormal_dir):
        for img_path in sorted(Path(test_abnormal_dir).glob('*_abnormal.png')):
            test_img_paths.append(str(img_path))
            test_gt_paths.append(0)   # 该数据集无 pixel-level GT
            test_labels.append(1)
            test_types.append(f'{category}_abnormal')

    return (train_img_paths, train_gt_paths, train_labels, train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)
