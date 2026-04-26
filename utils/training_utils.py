import random
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(TASK, root_dir, **kwargs):

    k_shot = kwargs['k_shot']
    dataset = kwargs['dataset']
    class_name = kwargs['class_name']
    train_site = kwargs.get('train_site', None)

    # 跨站点测试时，目录名包含训练站点信息
    if train_site and train_site != class_name:
        dir_class_name = f'{train_site}_to_{class_name}'
    else:
        dir_class_name = class_name

    # 对于 burst/dsss 等有 noise_level 的数据集，按 noise_level 区分目录
    # deceptive_signal 没有 noise_level 参数但目录结构中固定为 0db
    noise_level = kwargs.get('noise_level', None)
    if dataset == 'deceptive_signal':
        base_dir = os.path.join(root_dir, f'{dataset}', f'{dir_class_name}', '0db', f'k_{k_shot}')
    elif dataset == 'rf_open':
        # category already encodes signal_type + jsr (e.g. burst_m10db); no extra subdir
        base_dir = os.path.join(root_dir, f'{dataset}', f'{dir_class_name}', f'k_{k_shot}')
    elif noise_level:
        base_dir = os.path.join(root_dir, f'{dataset}', f'{dir_class_name}', f'{noise_level}', f'k_{k_shot}')
    else:
        base_dir = os.path.join(root_dir, f'{dataset}', f'{dir_class_name}', f'k_{k_shot}')

    csv_dir = os.path.join(base_dir, 'csv')
    check_dir = os.path.join(base_dir, 'checkpoint')
    img_dir = os.path.join(base_dir, 'imgs')

    csv_path = os.path.join(csv_dir, f"Seed_{kwargs['seed']}-results.csv")
    check_path = os.path.join(check_dir, f"{TASK}-Seed_{kwargs['seed']}-{class_name}-check_point.pt")

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return img_dir, csv_path, check_path
