import glob
import os


sample_classes = ['burst', 'stealthy']


SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample'))


def load_sample(category, k_shot):
    if k_shot < 1:
        raise ValueError(f'k_shot must be >= 1 for training, but got {k_shot}. Use --k-shot 1 or larger.')

    def collect_pngs(root_path):
        return sorted(glob.glob(os.path.join(root_path, '*.png')))

    def load_train_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.isdir(root_path):
            raise FileNotFoundError(f'Sample dataset split folder not found: {root_path}')

        scene_dirs = sorted([name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))])
        for scene_name in scene_dirs:
            scene_dir = os.path.join(root_path, scene_name)
            img_paths = collect_pngs(scene_dir)

            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend([0] * len(img_paths))
            tot_labels.extend([0] * len(img_paths))
            tot_types.extend(['good'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with sample data pairing!'

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def load_test_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.isdir(root_path):
            raise FileNotFoundError(f'Sample dataset split folder not found: {root_path}')

        db_dirs = sorted([name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))])
        for db_name in db_dirs:
            db_dir = os.path.join(root_path, db_name)
            scene_dirs = sorted([name for name in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, name))])

            for scene_name in scene_dirs:
                scene_dir = os.path.join(db_dir, scene_name)

                normal_dir = os.path.join(scene_dir, 'normal')
                abnormal_dir = os.path.join(scene_dir, 'abnormal')
                gt_dir = os.path.join(scene_dir, 'groundtruth')

                normal_paths = collect_pngs(normal_dir)
                for img_path in normal_paths:
                    img_tot_paths.append(img_path)
                    gt_tot_paths.append(0)
                    tot_labels.append(0)
                    tot_types.append(db_name)

                abnormal_paths = collect_pngs(abnormal_dir)
                for img_path in abnormal_paths:
                    img_name = os.path.basename(img_path)
                    gt_name = f'{os.path.splitext(img_name)[0]}_gt.png'
                    gt_path = os.path.join(gt_dir, gt_name)
                    img_tot_paths.append(img_path)
                    gt_tot_paths.append(gt_path)
                    tot_labels.append(1)
                    tot_types.append(db_name)

        assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with sample test data pairing!'

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def load_burst_train(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.isdir(root_path):
            raise FileNotFoundError(f'Sample dataset split folder not found: {root_path}')

        good_dir = os.path.join(root_path, 'good')
        img_paths = collect_pngs(good_dir) if os.path.isdir(good_dir) else collect_pngs(root_path)

        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))
        tot_labels.extend([0] * len(img_paths))
        tot_types.extend(['good'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with burst train data pairing!'

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def load_burst_test(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.isdir(root_path):
            raise FileNotFoundError(f'Sample dataset split folder not found: {root_path}')

        for type_name, label in [('good', 0), ('bad', 1)]:
            type_dir = os.path.join(root_path, type_name)
            img_paths = collect_pngs(type_dir) if os.path.isdir(type_dir) else []
            for img_path in img_paths:
                img_tot_paths.append(img_path)
                gt_tot_paths.append(0)
                tot_labels.append(label)
                tot_types.append(type_name)

        assert len(img_tot_paths) == len(gt_tot_paths), 'Something wrong with burst test data pairing!'

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in sample_classes

    category_root = os.path.join(SAMPLE_DIR, category)
    test_img_path = os.path.join(category_root, 'test')
    train_img_path = os.path.join(category_root, 'train')

    if category == 'burst':
        train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_burst_train(train_img_path)
        test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_burst_test(test_img_path)
    else:
        train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_train_phase(train_img_path)
        test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_test_phase(test_img_path)

    train_good_indices = [idx for idx, label in enumerate(train_tot_labels) if label == 0]
    selected_train_indices = train_good_indices[:min(k_shot, len(train_good_indices))]

    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_train_indices]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_train_indices]
    selected_train_tot_labels = [train_tot_labels[k] for k in selected_train_indices]
    selected_train_tot_types = [train_tot_types[k] for k in selected_train_indices]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)