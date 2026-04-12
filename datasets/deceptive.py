import os
from collections import OrderedDict
from pathlib import Path


deceptive_classes = ['deceptive']


DECEPTIVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DCASE2025_fbank'))


def _infer_label_from_path(path: str) -> tuple[int, str]:
    """从文件路径推断标签和类型"""
    lower_name = os.path.basename(path).lower()

    # 测试集：根据文件名判断
    if 'anomaly' in lower_name:
        return 1, 'anomaly'
    if 'normal' in lower_name:
        return 0, 'normal'

    # 训练集/补充集：默认为正常
    return 0, 'normal'


def _collect_phase(root_paths, include_anomaly: bool):
    """收集指定目录下的所有文件"""
    img_tot_paths = []
    gt_tot_paths = []
    tot_labels = []
    tot_types = []

    for root_path in root_paths:
        if not os.path.isdir(root_path):
            continue

        img_paths = sorted(Path(root_path).rglob('*.png'))
        for img_path in img_paths:
            label, img_type = _infer_label_from_path(str(img_path))
            if label == 1 and not include_anomaly:
                continue

            img_tot_paths.append(str(img_path))
            gt_tot_paths.append(0)
            tot_labels.append(label)
            tot_types.append(img_type)

    assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with data pairing!'

    return img_tot_paths, gt_tot_paths, tot_labels, tot_types


def _select_normal_samples_by_segments(img_paths, labels, max_segments):
    """
    按 segment 数量选择前 N 个 segment 用于训练

    文件名格式：section_00_source_train_normal_XXXX_noAttribute_seg_YY.png
    按 normal_XXXX 分组，选择前 max_segments 个 segment

    例如：max_segments=4000 选择前 4000 个 segment（约 500 个样本）
    """
    grouped_indices = OrderedDict()

    for idx, (img_path, label) in enumerate(zip(img_paths, labels)):
        if label != 0:
            continue

        stem = Path(img_path).stem
        # 提取 normal_XXXX 作为组名
        parts = stem.split('_')
        for i, part in enumerate(parts):
            if part == 'normal' and i + 1 < len(parts):
                group_key = f"normal_{parts[i + 1]}"
                break
        else:
            group_key = stem

        if group_key not in grouped_indices:
            grouped_indices[group_key] = []
        grouped_indices[group_key].append(idx)

    # 按顺序选择 segment，直到达到 max_segments
    selected_indices = []
    remaining = max_segments

    for group_key, indices in grouped_indices.items():
        # 每组最多取 8 个 segment（seg_00 到 seg_07）
        take = min(len(indices), remaining)
        selected_indices.extend(indices[:take])
        remaining -= take
        if remaining <= 0:
            break

    return selected_indices


def load_deceptive(category, k_shot):
    """
    加载 deceptive 数据集

    训练集：使用 DCASE2025_fbank/bearing/train/ 中的正常样本
    测试集：使用 DCASE2025_fbank/bearing/test/ 中的正常 + 异常样本

    k_shot: 选择前 k_shot 个 segment 用于训练
            例如 k_shot=4000 使用前 4000 个 segment（约 500 个样本）
            如果你想用 0-4000 频段，设置 k_shot=4000
    """
    if k_shot < 1:
        raise ValueError(f'k_shot must be >= 1 for training, but got {k_shot}. Use --k-shot 1 or larger.')

    # 数据目录
    train_root = os.path.join(DECEPTIVE_DIR, 'bearing', 'train')
    test_root = os.path.join(DECEPTIVE_DIR, 'bearing', 'test')

    # 训练集：只包含正常样本
    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = _collect_phase(
        [train_root],
        include_anomaly=False,
    )

    # 测试集：包含正常和异常
    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = _collect_phase(
        [test_root],
        include_anomaly=True,
    )

    # 选择前 k_shot 个 segment
    selected_train_indices = _select_normal_samples_by_segments(train_img_tot_paths, train_tot_labels, k_shot)

    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_train_indices]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_train_indices]
    selected_train_tot_labels = [train_tot_labels[k] for k in selected_train_indices]
    selected_train_tot_types = [train_tot_types[k] for k in selected_train_indices]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
