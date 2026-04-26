import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot, **kwargs):

        self.load_function = load_function
        self.phase = phase

        self.category = category
        self.extra_kwargs = kwargs

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, k_shot):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(
            self.category, k_shot, **self.extra_kwargs
        )
        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_h, orig_w = img.shape[:2]

        if gt == 0:
            gt = np.zeros([orig_h, orig_w])
        else:
            # 读取彩色 groundtruth，黄色=异常 (1)，紫色=正常 (0)
            gt_color = cv2.imread(gt, cv2.IMREAD_COLOR)
            # 黄色在 BGR 中约为 (0, 255, 255)
            # 阈值分割：B<50, G>200, R>200
            lower_yellow = np.array([0, 200, 200])
            upper_yellow = np.array([50, 255, 255])
            mask = cv2.inRange(gt_color, lower_yellow, upper_yellow)
            gt = mask

        # Cap large images at 1024; never upscale small images
        target_size = min(max(orig_h, orig_w), 1024)
        img = cv2.resize(img, (target_size, target_size))
        gt = cv2.resize(gt, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        return img, gt, label, img_name, img_type
