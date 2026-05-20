import os
import re
from pathlib import Path

from .rf_split_utils import load_rf_split_dataset


chirp_signal_classes = ['WeaponMuseum_spectrum', 'Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']


NORMAL_DIR = '/mnt/data/wangbei/data/datasets/normal'
CHIRP_SIGNAL_DIR = '/mnt/data/wangbei/data/datasets/chirp'


def _extract_time_range(filename: str) -> tuple:
    """从文件名提取时间范围
    例如：BinBo_chirp_m10db_t00000-04000_f100.00-101.50MHz_patch016_normal.png
    返回：(0, 4000)
    """
    match = re.search(r't(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


def _extract_freq(filename: str) -> str:
    """从文件名提取频段
    例如：BinBo_chirp_m10db_t00000-04000_f100.00-101.50MHz_patch016_normal.png
    返回：100.00-101.50
    """
    match = re.search(r'f(\d+\.\d+)-(\d+\.\d+)MHz', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def load_chirp_signal(category, k_shot, noise_level='m10db', freq=None, train_category=None,
                      split_mode='legacy', normal_train_ratio=0.75):
    """
    加载 chirp 信号数据集用于 chirp 信号检测（按场景或频段分开）

    数据结构:
    /mnt/data/wangbei/data/datasets/
    ├── normal/                         # 共用正常训练数据
    │   └── {category}/normal/          # 正常样本 (多个时间窗口)
    └── chirp/                          # Chirp 信号测试数据
        └── {category}/
            ├── normal/{noise_level}/           # 正常 chirp 信号 (测试)
            ├── abnormal/{noise_level}/         # 异常 chirp 信号 (测试)
            └── groundtruth/{noise_level}/      # Groundtruth 图片

    Args:
        category: 场景名称 (BinBo, CaoChang, ShiJianGuangChang, TiYuGuan)
        k_shot: 训练样本数量（从 t00000-04000 时间段中选取）
        noise_level: 噪声水平 (m10db, m20db, m30db)，默认 m10db
        freq: 频段 (如 "100.00-101.50")，如果为 None 则加载该场景的所有频段

    Returns:
        train_data: (img_paths, gt_paths, labels, types)
        test_data: (img_paths, gt_paths, labels, types)
    """
    if split_mode == 'normal_75_25':
        return load_rf_split_dataset(
            CHIRP_SIGNAL_DIR, category, noise_level=noise_level, freq=freq,
            normal_train_ratio=normal_train_ratio
        )

    train_cat = train_category if train_category is not None else category
    train_root = os.path.join(NORMAL_DIR, train_cat)
    test_normal_root = os.path.join(CHIRP_SIGNAL_DIR, category, 'normal', noise_level)
    test_abnormal_root = os.path.join(CHIRP_SIGNAL_DIR, category, 'abnormal', noise_level)
    groundtruth_root = os.path.join(CHIRP_SIGNAL_DIR, category, 'groundtruth', noise_level)

    # 收集训练集 (只包含正常样本，t00000-04000 时间段，按 k_shot 采样)
    train_img_paths = []
    train_labels = []
    train_types = []

    if os.path.isdir(train_root):
        for img_path in sorted(Path(train_root).glob('*.png')):
            file_time = _extract_time_range(img_path.name)
            if file_time != (0, 4000):
                continue
            # 如果指定了频段，只加载该频段的样本
            if freq is not None:
                file_freq = _extract_freq(img_path.name)
                if file_freq != freq:
                    continue
            train_img_paths.append(str(img_path))
            train_labels.append(0)
            train_types.append(f'{category}_normal')

        # k_shot 采样：从前 k_shot 个样本中选取
        if k_shot > 0 and len(train_img_paths) > k_shot:
            train_img_paths = train_img_paths[:k_shot]
            train_labels = train_labels[:k_shot]
            train_types = train_types[:k_shot]

    # 收集测试集
    test_img_paths = []
    test_gt_paths = []
    test_labels = []
    test_types = []

    # 正常测试样本
    if os.path.isdir(test_normal_root):
        for img_path in sorted(Path(test_normal_root).glob('*.png')):
            # 如果指定了频段，只加载该频段的样本
            if freq is not None:
                file_freq = _extract_freq(img_path.name)
                if file_freq != freq:
                    continue
            test_img_paths.append(str(img_path))
            img_name = img_path.name.replace('_normal.png', '_groundtruth.png')
            gt_path = os.path.join(groundtruth_root, img_name)
            test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
            test_labels.append(0)
            test_types.append(f'{category}_normal')

    # 异常测试样本
    if os.path.isdir(test_abnormal_root):
        for img_path in sorted(Path(test_abnormal_root).glob('*.png')):
            # 如果指定了频段，只加载该频段的样本
            if freq is not None:
                file_freq = _extract_freq(img_path.name)
                if file_freq != freq:
                    continue
            test_img_paths.append(str(img_path))
            img_name = img_path.name.replace('_abnormal.png', '_groundtruth.png')
            gt_path = os.path.join(groundtruth_root, img_name)
            test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
            test_labels.append(1)
            test_types.append(f'{category}_abnormal')

    # 训练集 GT 设为 0（不需要）
    selected_train_gt_paths = [0] * len(train_img_paths)

    return (train_img_paths, selected_train_gt_paths, train_labels, train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)
