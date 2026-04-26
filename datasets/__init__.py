import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes
from .spectrum import load_spectrum, spectrum_classes
from .sample import load_sample, sample_classes
from .deceptive_signal import load_deceptive_signal, deceptive_signal_classes
from .burst_signal import load_burst_signal, burst_signal_classes
from .dsss_signal import load_dsss_signal, dsss_classes
from .chirp_signal import load_chirp_signal, chirp_signal_classes
from .rf_open_dataset import load_rf_open, rf_open_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
    'spectrum': load_spectrum,
    'sample': load_sample,
    'deceptive_signal': load_deceptive_signal,
    'burst_signal': load_burst_signal,
    'dsss_signal': load_dsss_signal,
    'chirp_signal': load_chirp_signal,
    'rf_open': load_rf_open,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
    'spectrum': spectrum_classes,
    'sample': sample_classes,
    'deceptive_signal': deceptive_signal_classes,
    'burst_signal': burst_signal_classes,
    'dsss_signal': dsss_classes,
    'chirp_signal': chirp_signal_classes,
    'rf_open': rf_open_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):

    # 提取 burst_signal 额外的参数
    extra_kwargs = {}
    if kwargs.get('dataset') == 'burst_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
        if kwargs.get('train_site'):
            extra_kwargs['train_category'] = kwargs.get('train_site')
    elif kwargs.get('dataset') == 'dsss_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
        if kwargs.get('train_site'):
            extra_kwargs['train_category'] = kwargs.get('train_site')
    elif kwargs.get('dataset') == 'chirp_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
        if kwargs.get('train_site'):
            extra_kwargs['train_category'] = kwargs.get('train_site')
    elif kwargs.get('dataset') == 'rf_open':
        pass  # category 已编码 signal_type + noise_level，无需额外 kwargs
    elif kwargs.get('dataset') == 'deceptive_signal':
        extra_kwargs['freq'] = kwargs.get('freq', None)
        if kwargs.get('train_site'):
            extra_kwargs['train_category'] = kwargs.get('train_site')

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        **extra_kwargs
    )

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    # debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    # # logger.info(debug_str)

    return data_loader, dataset_inst