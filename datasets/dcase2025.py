import glob
import os
from collections import OrderedDict
from pathlib import Path


dcase2025_classes = ['bearing']


DCASE2025_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DCASE2025_fbank'))


def _infer_label_from_path(path: str) -> tuple[int, str]:
    lower_path = path.lower()
    if 'supplemental' in lower_path:
        return 0, 'normal'

    lower_name = os.path.basename(path).lower()
    if 'anomaly' in lower_name:
        return 1, 'anomaly'
    if 'normal' in lower_name:
        return 0, 'normal'
    raise ValueError(f'Cannot infer label from file name: {path}')


def _collect_phase(root_paths, include_anomaly: bool):
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

    assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with DCASE2025 data pairing!'

    return img_tot_paths, gt_tot_paths, tot_labels, tot_types


def _select_normal_groups(img_paths, labels, k_shot):
    grouped_indices = OrderedDict()

    for idx, (img_path, label) in enumerate(zip(img_paths, labels)):
        if label != 0:
            continue

        stem = Path(img_path).stem
        base_stem = stem.rsplit('_seg_', 1)[0]
        group_key = (str(Path(img_path).parent), base_stem)
        if group_key not in grouped_indices:
            grouped_indices[group_key] = []
        grouped_indices[group_key].append(idx)

    selected_indices = []
    for group_key in list(grouped_indices.keys())[:k_shot]:
        selected_indices.extend(grouped_indices[group_key])

    return selected_indices


def load_dcase2025(category, k_shot):
    if k_shot < 1:
        raise ValueError(f'k_shot must be >= 1 for training, but got {k_shot}. Use --k-shot 1 or larger.')

    assert category in dcase2025_classes

    train_root = os.path.join(DCASE2025_DIR, category, 'train')
    test_root = os.path.join(DCASE2025_DIR, category, 'test')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = _collect_phase(
        [train_root],
        include_anomaly=False,
    )
    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = _collect_phase(
        [test_root],
        include_anomaly=True,
    )

    selected_train_indices = _select_normal_groups(train_img_tot_paths, train_tot_labels, k_shot)

    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_train_indices]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_train_indices]
    selected_train_tot_labels = [train_tot_labels[k] for k in selected_train_indices]
    selected_train_tot_types = [train_tot_types[k] for k in selected_train_indices]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)