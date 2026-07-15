import os
import re
from pathlib import Path

pulse_signal_classes = [
    'WeaponMuseum_spectrum',
    'Playground_spectrum',
    'TimeSquare_spectrum',
    'Gymnasium_spectrum',
]


PULSE_SIGNAL_DIR = '/mnt/data/wangbei/data/datasets/pulse'
NORMAL_DIR = '/mnt/data/wangbei/data/datasets/normal'


def _extract_time_range(filename: str) -> tuple:
    match = re.search(r't(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


def _extract_freq(filename: str):
    match = re.search(r'f(\d+\.\d+)-(\d+\.\d+)MHz', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def load_pulse_signal(category, k_shot, noise_level='m20db', freq=None, train_category=None):
    train_cat = train_category if train_category is not None else category
    train_root = os.path.join(NORMAL_DIR, train_cat)
    test_normal_root = os.path.join(PULSE_SIGNAL_DIR, category, 'normal', noise_level)
    test_abnormal_root = os.path.join(PULSE_SIGNAL_DIR, category, 'abnormal', noise_level)
    groundtruth_root = os.path.join(PULSE_SIGNAL_DIR, category, 'groundtruth', noise_level)

    train_img_paths = []
    train_labels = []
    train_types = []
    if os.path.isdir(train_root):
        for img_path in sorted(Path(train_root).glob('*.png')):
            if _extract_time_range(img_path.name) != (0, 4000):
                continue
            if freq is not None and _extract_freq(img_path.name) != freq:
                continue
            train_img_paths.append(str(img_path))
            train_labels.append(0)
            train_types.append(f'{category}_normal')
        if k_shot > 0 and len(train_img_paths) > k_shot:
            train_img_paths = train_img_paths[:k_shot]
            train_labels = train_labels[:k_shot]
            train_types = train_types[:k_shot]

    test_img_paths = []
    test_gt_paths = []
    test_labels = []
    test_types = []
    if os.path.isdir(test_normal_root):
        for img_path in sorted(Path(test_normal_root).glob('*.png')):
            if freq is not None and _extract_freq(img_path.name) != freq:
                continue
            test_img_paths.append(str(img_path))
            img_name = img_path.name.replace('_normal.png', '_groundtruth.png')
            gt_path = os.path.join(groundtruth_root, img_name)
            test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
            test_labels.append(0)
            test_types.append(f'{category}_normal')

    if os.path.isdir(test_abnormal_root):
        for img_path in sorted(Path(test_abnormal_root).glob('*.png')):
            if freq is not None and _extract_freq(img_path.name) != freq:
                continue
            test_img_paths.append(str(img_path))
            img_name = img_path.name.replace('_abnormal.png', '_groundtruth.png')
            gt_path = os.path.join(groundtruth_root, img_name)
            test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
            test_labels.append(1)
            test_types.append(f'{category}_abnormal')

    train_gt_paths = [0] * len(train_img_paths)
    return (train_img_paths, train_gt_paths, train_labels, train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)
