from .rf_split_utils import load_rf_split_dataset


wideband_pulse_classes = [
    'WeaponMuseum_spectrum',
    'Playground_spectrum',
    'TimeSquare_spectrum',
    'Gymnasium_spectrum',
]


WIDEBAND_PULSE_DIR = '/mnt/data/wangbei/data/datasets/wideband_pulse'


def load_wideband_pulse(category, k_shot, noise_level='m20db', freq=None, train_category=None,
                        split_mode='legacy', normal_train_ratio=0.75):
    if split_mode != 'normal_75_25':
        raise ValueError(
            'wideband_pulse currently supports --split-mode normal_75_25 only.'
        )

    return load_rf_split_dataset(
        WIDEBAND_PULSE_DIR,
        category,
        noise_level=noise_level,
        freq=freq,
        normal_train_ratio=normal_train_ratio,
    )
