import glob
import os


spectrum_classes = ['16QAM', 'CHIRP', 'GMSK', 'QPSK']


SPECTRUM_DIR = './datasets/spectrum'


def load_spectrum(category, k_shot):
    if k_shot < 1:
        raise ValueError(f'k_shot must be >= 1 for training, but got {k_shot}. Use --k-shot 1 or larger.')

    def load_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = sorted(os.listdir(root_path))

        for defect_type in defect_types:
            defect_dir = os.path.join(root_path, defect_type)
            if not os.path.isdir(defect_dir):
                continue

            img_paths = sorted(glob.glob(os.path.join(defect_dir, '*.png')))

            if defect_type == 'good':
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with spectrum test data pairing!'

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in spectrum_classes

    test_img_path = os.path.join(SPECTRUM_DIR, category, 'test')
    train_img_path = os.path.join(SPECTRUM_DIR, category, 'train')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_phase(train_img_path)
    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_phase(test_img_path)

    # Few-shot training: keep the first k_shot normal samples in a deterministic order.
    train_good_indices = [idx for idx, label in enumerate(train_tot_labels) if label == 0]
    selected_train_indices = train_good_indices[:k_shot]

    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_train_indices]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_train_indices]
    selected_train_tot_labels = [train_tot_labels[k] for k in selected_train_indices]
    selected_train_tot_types = [train_tot_types[k] for k in selected_train_indices]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)