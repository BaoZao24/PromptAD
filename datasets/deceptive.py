import os
import re
from pathlib import Path


deceptive_classes = ['deceptive']


DECEPTIVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample', 'deceptive'))


def _extract_time_range(filename: str) -> tuple:
    """从文件名提取时间范围
    例如：BinBo_burst_t00000-04000_f100.00-101.50MHz_patch016_normal.png
    返回：(0, 4000)
    """
    match = re.search(r't(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


def _find_groundtruth_path(img_path: str, test_scene_dir: Path) -> str:
    """
    根据图像路径查找对应的 groundtruth 文件

    例如：
    - 输入：.../test/BinBo/abnormal/0db/BinBo_stealthy_0db_t00000-04000_f96.80-98.30MHz_patch012_abnormal.png
    - 输出：.../test/BinBo/groundtruth/0db/BinBo_stealthy_0db_t00000-04000_f96.80-98.30MHz_patch012_groundtruth.png
    """
    img_name = os.path.basename(img_path)

    # 提取文件名主体（去掉 _abnormal 或 _normal 后缀）
    if '_abnormal.png' in img_name:
        base_name = img_name.replace('_abnormal.png', '_groundtruth.png')
    elif '_normal.png' in img_name:
        base_name = img_name.replace('_normal.png', '_groundtruth.png')
    else:
        base_name = img_name

    # 获取子目录（如 0db）- 跳过 abnormal/normal 目录
    rel_path = os.path.relpath(str(img_path), str(test_scene_dir))
    parts = rel_path.split(os.sep)
    # 过滤掉 abnormal/normal 目录，只保留子目录（如 0db）
    subdir_parts = [p for p in parts[:-1] if p not in ('abnormal', 'normal')]
    subdir = os.sep.join(subdir_parts) if subdir_parts else ''

    # 构建 groundtruth 路径
    if subdir:
        gt_path = test_scene_dir / 'groundtruth' / subdir / base_name
    else:
        gt_path = test_scene_dir / 'groundtruth' / base_name

    if gt_path.exists():
        return str(gt_path)

    return None


def load_deceptive(category, k_shot):
    """
    加载 deceptive 数据集用于欺骗信号检测

    数据结构:
    datasets/sample/deceptive/
    ├── train/
    │   ├── BinBo/normal/      # 正常训练样本 (t00000-04000, t02000-06000, 等 8 个时间段)
    │   ├── CaoChang/normal/
    │   ├── ShiJianGuangChang/normal/
    │   └── TiYuGuan/normal/
    └── test/
        ├── BinBo/normal/      # 正常测试样本
        ├── BinBo/abnormal/    # 异常测试样本 (欺骗信号)
        ├── CaoChang/...
        ├── ShiJianGuangChang/...
        └── TiYuGuan/...

    训练集：只使用 train/ 中各场景的 normal 样本
    测试集：使用 test/ 中各场景的 normal + abnormal 样本

    Args:
        category: 数据集类别名称（未使用，保持 API 一致）
        k_shot: 选择训练样本的策略:
                - k_shot=0 或使用 0-4000 时间段的样本 (约 912 个样本)
                - k_shot>0 且指定使用所有时间段：使用所有训练样本

    注意：文件名格式为 {scene}_{type}_tXXXXX-YYYYY_fAA.AA-BB.BBMHz_patchNNN_normal.png
    """
    train_root = os.path.join(DECEPTIVE_DIR, 'train')
    test_root = os.path.join(DECEPTIVE_DIR, 'test')

    # 收集训练集 (只包含正常样本)
    train_img_paths = []
    train_labels = []
    train_types = []

    scene_dirs = sorted([d for d in Path(train_root).iterdir() if d.is_dir()])
    for scene_dir in scene_dirs:
        normal_dir = scene_dir / 'normal'
        if normal_dir.exists():
            for img_path in sorted(normal_dir.glob('*.png')):
                # 只选择 t00000-04000 时间段的样本用于训练
                file_time = _extract_time_range(img_path.name)
                if file_time != (0, 4000):
                    continue

                train_img_paths.append(str(img_path))
                train_labels.append(0)  # normal
                train_types.append(f'{scene_dir.name}_normal')

    # 收集测试集 (包含正常和异常)
    test_img_paths = []
    test_gt_paths = []
    test_labels = []
    test_types = []

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        test_scene_dir = Path(test_root) / scene_name

        # 正常测试样本 (递归搜索所有子目录)
        if (test_scene_dir / 'normal').exists():
            for img_path in sorted((test_scene_dir / 'normal').rglob('*.png')):
                test_img_paths.append(str(img_path))
                gt_path = _find_groundtruth_path(str(img_path), test_scene_dir)
                test_gt_paths.append(gt_path if gt_path else 0)
                test_labels.append(0)
                test_types.append(f'{scene_name}_normal')

        # 异常测试样本 (递归搜索所有子目录)
        if (test_scene_dir / 'abnormal').exists():
            for img_path in sorted((test_scene_dir / 'abnormal').rglob('*.png')):
                test_img_paths.append(str(img_path))
                gt_path = _find_groundtruth_path(str(img_path), test_scene_dir)
                test_gt_paths.append(gt_path if gt_path else 0)
                test_labels.append(1)
                test_types.append(f'{scene_name}_abnormal')

    # k_shot 参数保留但不使用（所有 t00000-04000 样本都用于训练）
    selected_train_img_paths = train_img_paths
    selected_train_labels = train_labels
    selected_train_types = train_types

    # 创建 dummy ground truth 路径 (所有都是 0)
    selected_train_gt_paths = [0] * len(selected_train_img_paths)

    return (selected_train_img_paths, selected_train_gt_paths, selected_train_labels, selected_train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)
