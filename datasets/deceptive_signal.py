import os
import re
from pathlib import Path


deceptive_signal_classes = ['WeaponMuseum_spectrum', 'Playground_spectrum', 'TimeSquare_spectrum', 'Gymnasium_spectrum']


NORMAL_DIR = '/mnt/data/wangbei/data/datasets/normal'
DECEPTIVE_SIGNAL_DIR = '/mnt/data/wangbei/data/datasets/deceptive'


def _extract_time_range(filename: str) -> tuple:
    """从文件名提取时间范围
    例如：BinBo_normal_t00000-04000_f100.00-101.50MHz_patch016_normal.png
    返回：(0, 4000)
    """
    match = re.search(r't(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


def _extract_freq(filename: str) -> str:
    """从文件名提取频段
    例如：BinBo_normal_t00000-04000_f100.00-101.50MHz_patch016_normal.png
    返回：100.00-101.50
    """
    match = re.search(r'f(\d+\.\d+)-(\d+\.\d+)MHz', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def load_deceptive_signal(category, k_shot, freq=None, train_category=None):
    """
    加载 deceptive(stealthy) 信号数据集用于欺骗信号检测（按场景或频段分开）

    数据结构:
    /mnt/data/wangbei/data/datasets/
    ├── normal/                         # 共用正常训练数据
    │   └── {category}/normal/          # 正常样本 (多个时间窗口)
    └── deceptive/                      # Deceptive 信号测试数据
        └── {category}/
            ├── normal/0db/             # 正常 stealthy 信号 (测试)
            ├── abnormal/0db/           # 异常 stealthy/deceptive 信号 (测试)
            └── groundtruth/0db/        # Groundtruth 图片

    Args:
        category: 场景名称 (BinBo, CaoChang, ShiJianGuangChang, TiYuGuan)
        k_shot: 训练样本数量（从 t00000-04000 时间段中选取）
        freq: 频段 (如 "100.00-101.50")，如果为 None 则加载该场景的所有频段

    Returns:
        train_data: (img_paths, gt_paths, labels, types)
        test_data: (img_paths, gt_paths, labels, types)
    """
    train_cat = train_category if train_category is not None else category
    train_root = os.path.join(NORMAL_DIR, train_cat)
    test_normal_root = os.path.join(DECEPTIVE_SIGNAL_DIR, category, 'normal', '0db')
    test_abnormal_root = os.path.join(DECEPTIVE_SIGNAL_DIR, category, 'abnormal', '0db')
    groundtruth_root = os.path.join(DECEPTIVE_SIGNAL_DIR, category, 'groundtruth', '0db')

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
            # Groundtruth 路径：/groundtruth/0db/{name}_groundtruth.png
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
            # Groundtruth 路径：/groundtruth/0db/{name}_groundtruth.png
            img_name = img_path.name.replace('_abnormal.png', '_groundtruth.png')
            gt_path = os.path.join(groundtruth_root, img_name)
            test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
            test_labels.append(1)
            test_types.append(f'{category}_abnormal')

    # 训练集 GT 设为 0（不需要）
    selected_train_gt_paths = [0] * len(train_img_paths)

    return (train_img_paths, selected_train_gt_paths, train_labels, train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)
