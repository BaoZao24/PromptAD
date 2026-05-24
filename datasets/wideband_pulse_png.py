import os
from pathlib import Path

from .rf_split_utils import _filter_by_freq, _split_normal_paths


wideband_pulse_png_classes = ['wideband_pulse']


WIDEBAND_PULSE_PNG_DIR = '/mnt/data/wangbei/data/RF_SPE_PNG/wideband_pulse'


def load_wideband_pulse_png(category, k_shot, noise_level='m20db', freq=None, train_category=None,
                            split_mode='legacy', normal_train_ratio=0.75):
    if category != 'wideband_pulse':
        raise ValueError(
            f"wideband_pulse_png only supports class_name='wideband_pulse', got {category!r}."
        )
    if split_mode != 'normal_75_25':
        raise ValueError(
            'wideband_pulse_png currently supports --split-mode normal_75_25 only.'
        )

    normal_root = os.path.join(WIDEBAND_PULSE_PNG_DIR, 'normal', noise_level)
    abnormal_root = os.path.join(WIDEBAND_PULSE_PNG_DIR, 'abnormal', noise_level)
    gt_root = os.path.join(WIDEBAND_PULSE_PNG_DIR, 'groundtruth', noise_level)

    normal_paths = []
    if os.path.isdir(normal_root):
        normal_paths = sorted(str(p) for p in Path(normal_root).glob('*.png'))
        normal_paths = _filter_by_freq(normal_paths, freq)

    abnormal_paths = []
    if os.path.isdir(abnormal_root):
        abnormal_paths = sorted(str(p) for p in Path(abnormal_root).glob('*.png'))
        abnormal_paths = _filter_by_freq(abnormal_paths, freq)

    train_normal_paths, test_normal_paths = _split_normal_paths(normal_paths, normal_train_ratio)

    train_img_paths = list(train_normal_paths)
    train_gt_paths = [0] * len(train_img_paths)
    train_labels = [0] * len(train_img_paths)
    train_types = [f'{category}_normal' for _ in train_img_paths]

    test_img_paths = list(test_normal_paths)
    test_gt_paths = [0] * len(test_img_paths)
    test_labels = [0] * len(test_img_paths)
    test_types = [f'{category}_normal' for _ in test_img_paths]

    for img_path in abnormal_paths:
        img_name = os.path.basename(img_path).replace('_abnormal.png', '_groundtruth.png')
        gt_path = os.path.join(gt_root, img_name)
        test_img_paths.append(img_path)
        test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
        test_labels.append(1)
        test_types.append(f'{category}_abnormal')

    return (train_img_paths, train_gt_paths, train_labels, train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)
