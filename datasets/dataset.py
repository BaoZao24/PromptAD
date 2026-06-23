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
        # 可选: 把 model.transform 注入到 __getitem__, 让重 CPU 的 morph_fusion 等
        # 预处理在 DataLoader worker 进程里跑, 不再阻塞主进程/GPU.
        # 注入后 __getitem__ 返回 (tensor[C,H,W], mask[H,W], label, name, type),
        # 否则保持向后兼容: 返回 (ndarray[H,W,3], mask, label, name, type).
        self._sample_transform = None

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot)  # self.labels => good : 0, anomaly : 1

    def set_sample_transform(self, transform):
        """注入图像预处理 callable (e.g. model.transform). 注入后 __getitem__ 直接出 tensor.

        Compose 是 picklable 的, 可安全在 spawn 模式 worker 中使用; fork 模式下
        worker 继承父进程的 transform 实例.
        """
        self._sample_transform = transform

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
            gt = np.zeros([orig_h, orig_w], dtype=np.uint8)
        else:
            gt_color = cv2.imread(gt, cv2.IMREAD_COLOR)
            if gt_color is None:
                gt = np.zeros([orig_h, orig_w], dtype=np.uint8)
            else:
                # Backward compatible GT decoding:
                # older masks mark anomalies in yellow; redone RF masks are binary white-on-black.
                lower_yellow = np.array([0, 200, 200])
                upper_yellow = np.array([50, 255, 255])
                yellow_mask = cv2.inRange(gt_color, lower_yellow, upper_yellow)
                if np.any(yellow_mask):
                    gt = yellow_mask
                else:
                    gt_gray = cv2.cvtColor(gt_color, cv2.COLOR_BGR2GRAY)
                    gt = ((gt_gray > 0).astype(np.uint8) * 255)

        # Cap large images at 1024; never upscale small images
        target_size = min(max(orig_h, orig_w), 1024)
        img = cv2.resize(img, (target_size, target_size))
        gt = cv2.resize(gt, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        if self._sample_transform is not None:
            # In-worker preprocessing path: 减轻主进程压力, 让 GPU 不再饿等数据.
            # transform 期望 PIL Image (RGB). cv2.imread 出来是 BGR.
            from PIL import Image as _PILImage
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_out = self._sample_transform(_PILImage.fromarray(img_rgb))
            return img_out, gt, label, img_name, img_type

        return img, gt, label, img_name, img_type
