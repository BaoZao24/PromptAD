import os
from pathlib import Path


def _extract_freq(filename: str):
    parts = filename.split('_f')
    if len(parts) < 2:
        return None
    tail = parts[-1]
    if 'MHz' not in tail:
        return None
    return tail.split('MHz', 1)[0]


def _filter_by_freq(img_paths, freq):
    if freq is None:
        return img_paths
    filtered = []
    for img_path in img_paths:
        if _extract_freq(Path(img_path).name) == freq:
            filtered.append(img_path)
    return filtered


def _split_normal_paths(normal_paths, normal_train_ratio):
    if not normal_paths:
        return [], []

    if normal_train_ratio is None:
        normal_train_ratio = 0.75

    n_total = len(normal_paths)
    if n_total == 1:
        return normal_paths[:1], []

    n_train = int(n_total * normal_train_ratio)
    n_train = max(1, min(n_train, n_total - 1))
    return normal_paths[:n_train], normal_paths[n_train:]


def load_rf_split_dataset(dataset_root, category, noise_level='m10db', freq=None, normal_train_ratio=0.75):
    """
    频谱异常检测的小批量实验切分：
    - 训练：normal/{noise_level} 中前 3/4 的正常图片
    - 测试：剩余 1/4 的正常图片 + abnormal/{noise_level} 的全部异常图片

    这里不再使用共用 normal 目录，也不再引入 train_site 迁移。
    """
    normal_root = os.path.join(dataset_root, category, 'normal', noise_level)
    abnormal_root = os.path.join(dataset_root, category, 'abnormal', noise_level)
    gt_root = os.path.join(dataset_root, category, 'groundtruth', noise_level)

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
