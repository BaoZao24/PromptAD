import os
from pathlib import Path

rf_spe_png_classes = ['burst', 'chirp', 'dsss', 'wideband_pulse']


RF_SPE_PNG_DIR = '/mnt/data/wangbei/data/RF_SPE_PNG'
RF_PUBLIC_NORMAL_DIR = '/mnt/data/wangbei/data/RF_SPE_PNG/RF_Spectrum_Public_Dataset'


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
    return [img_path for img_path in img_paths if _extract_freq(Path(img_path).name) == freq]


def _list_public_record_dirs():
    root = Path(RF_PUBLIC_NORMAL_DIR)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith('MeasRes_')
    )


def _load_public_normal_paths(k_shot, freq=None):
    record_dirs = _list_public_record_dirs()
    selected_train_dirs = record_dirs[:k_shot]
    selected_test_dirs = record_dirs[k_shot:]

    def collect_pngs(dirs):
        paths = []
        for d in dirs:
            paths.extend(sorted(str(p) for p in d.glob('*.png')))
        return _filter_by_freq(paths, freq)

    train_img_paths = collect_pngs(selected_train_dirs)
    test_normal_paths = collect_pngs(selected_test_dirs)
    return train_img_paths, test_normal_paths


def _load_with_public_normal(category, k_shot, noise_level, freq=None):
    train_img_paths, test_normal_paths = _load_public_normal_paths(k_shot, freq=freq)

    train_gt_paths = [0] * len(train_img_paths)
    train_labels = [0] * len(train_img_paths)
    train_types = [f'{category}_{noise_level}_train_normal' for _ in train_img_paths]

    test_img_paths = list(test_normal_paths)
    test_gt_paths = [0] * len(test_img_paths)
    test_labels = [0] * len(test_img_paths)
    test_types = [f'{category}_{noise_level}_normal' for _ in test_img_paths]

    abnormal_root = os.path.join(RF_SPE_PNG_DIR, category, 'abnormal', noise_level)
    gt_root = os.path.join(RF_SPE_PNG_DIR, category, 'groundtruth', noise_level)
    abnormal_paths = []
    if os.path.isdir(abnormal_root):
        abnormal_paths = sorted(str(p) for p in Path(abnormal_root).glob('*.png'))
        abnormal_paths = _filter_by_freq(abnormal_paths, freq)

    for img_path in abnormal_paths:
        img_name = os.path.basename(img_path).replace('_abnormal.png', '_groundtruth.png')
        gt_path = os.path.join(gt_root, img_name)
        test_img_paths.append(img_path)
        test_gt_paths.append(gt_path if os.path.exists(gt_path) else 0)
        test_labels.append(1)
        test_types.append(f'{category}_{noise_level}_abnormal')

    return (train_img_paths, train_gt_paths, train_labels, train_types),            (test_img_paths, test_gt_paths, test_labels, test_types)


def load_rf_spe_png(category, k_shot, noise_level='m10db', freq=None, train_category=None):
    if category not in rf_spe_png_classes:
        raise ValueError(
            f"rf_spe_png supports {rf_spe_png_classes}, got {category!r}."
        )
    return _load_with_public_normal(category, k_shot, noise_level, freq=freq)
