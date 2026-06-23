"""RF 跨库 (public source -> project target) 的 target 端池化 loader.

只在 `docs/PromptAD_adapter_experiment_plan.md` 描述的正式 RF 跨库实验里用。
约束（参考文档 §2 / §5）：
    - train split 只暴露 target 正常样本, 用于目标域正常特征库/normal adaptation;
    - train split 不暴露 target 异常样本和 mask, test split 用于最终评估;
    - 把 burst/chirp/dsss 三个 signal-type 数据集对应的 4 个 scene 池化到一起,
      使得文档 §7 要求的 burst_px / burst_sp 等指标按 signal-type 统计;
    - 不允许 burst<->chirp<->dsss 互转, 因此 category 只接受 'burst'/'chirp'/'dsss'.
"""

from .burst_signal import load_burst_signal, burst_signal_classes
from .chirp_signal import load_chirp_signal, chirp_signal_classes
from .dsss_signal import load_dsss_signal, dsss_classes


rf_target_test_pool_classes = ['burst', 'chirp', 'dsss']


_SIGNAL_LOADERS = {
    'burst': (load_burst_signal, burst_signal_classes),
    'chirp': (load_chirp_signal, chirp_signal_classes),
    'dsss': (load_dsss_signal, dsss_classes),
}


def _empty_split():
    return ([], [], [], [])


def load_rf_target_test_pool(
    category,
    k_shot,
    noise_level='m10db',
    freq=None,
    train_category=None,
    split_mode='normal_75_25',
    normal_train_ratio=0.75,
):
    """读取 4 个 scene, 返回 target train normal 池和 target test 池.

    每个样本的 type 字段会保留 scene 信息, 形如 'burst_Gymnasium_spectrum_normal',
    便于事后按 scene 拆分指标.
    """
    if category not in rf_target_test_pool_classes:
        raise ValueError(
            f"rf_target_test_pool only supports {rf_target_test_pool_classes}, got {category!r}."
        )

    if split_mode != 'normal_75_25':
        raise ValueError(
            "rf_target_test_pool requires --split-mode normal_75_25 "
            "to keep the target test split deterministic."
        )

    loader_fn, scene_list = _SIGNAL_LOADERS[category]

    pooled_train_imgs, pooled_train_gts, pooled_train_labels, pooled_train_types = [], [], [], []
    pooled_imgs, pooled_gts, pooled_labels, pooled_types = [], [], [], []
    for scene in scene_list:
        # k_shot 透传仅用于 legacy split, 这里强制 normal_75_25.
        train_data, test_data = loader_fn(
            category=scene,
            k_shot=k_shot,
            noise_level=noise_level,
            freq=freq,
            train_category=train_category,
            split_mode=split_mode,
            normal_train_ratio=normal_train_ratio,
        )
        scene_train_imgs, scene_train_gts, scene_train_labels, scene_train_types = train_data
        for img, gt, label, stype in zip(scene_train_imgs, scene_train_gts, scene_train_labels, scene_train_types):
            if int(label) != 0:
                continue
            pooled_train_imgs.append(img)
            pooled_train_gts.append(gt)
            pooled_train_labels.append(label)
            pooled_train_types.append(f"{category}_{scene}_target_train_{stype.split('_', 1)[-1]}")

        scene_imgs, scene_gts, scene_labels, scene_types = test_data
        pooled_imgs.extend(scene_imgs)
        pooled_gts.extend(scene_gts)
        pooled_labels.extend(scene_labels)
        # 把 scene 名称编进 type, 例如 'burst_Gymnasium_spectrum_abnormal'
        pooled_types.extend(
            [f"{category}_{scene}_{stype.split('_', 1)[-1]}" for stype in scene_types]
        )

    train_data = (pooled_train_imgs, pooled_train_gts, pooled_train_labels, pooled_train_types)
    test_data = (pooled_imgs, pooled_gts, pooled_labels, pooled_types)
    if not pooled_train_imgs:
        return _empty_split(), test_data
    return train_data, test_data
